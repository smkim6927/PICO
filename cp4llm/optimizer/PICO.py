# optimizer/PICO.py
import math
import torch
from torch.optim.optimizer import Optimizer


class PICO(Optimizer):
    """
    Paper-faithful PICO (matches Alg.1 GateU, Alg.2 SpecFlag, Alg.3 PICO).

    Implementation notes (consistent with notation table and pseudocode):
      - θ kept as fp32 master copy in optimizer state (one-time allocation).
      - avgU EMA kept as fp32 in state.
      - GateU uses u_max = ||avgU||_∞ + ε (avgU, *not* the bias-corrected one),
        and u = σ(avgU_hat / u_max). Fused into:
            u = σ(avgU / ((1-β_u^k + ε) · u_max))
        The (+ε) in the bias-correction is a safety guard for β=1.0 edge cases;
        with β<1 and k≥1, it has no numerical effect.
      - SpecFlag: ν is EMA of (x - μ_{c-1})²; z-score uses previous stats with
        stabilized variance ν̃ = max(ν, ε).
      - Group pause π only on check steps (k mod f == 0).
      - When s=1 or π=1, noise sampling is skipped (m^(k) ≡ 0).
      - Bugfix: μ_prev cloned before in-place EMA update so that ν uses true
        (x - μ_{c-1})², not (x - μ_c)².
    """

    def __init__(
        self,
        params,
        lr=2e-5,
        weight_decay=0.01,
        beta_utility=0.999,
        sigma=0.001,                # σ0 (base noise scale)
        spectral_update_freq=10,    # f (check period)
        power_iterations=1,         # K (power iterations per check)
        spec_c0=10,                 # c_0: warm-up checks before activation
        spec_r=30,                  # r: anneal length (in checks)
        spec_tau_start=6.0,         # τ_start
        spec_tau_end=3.0,           # τ_end
        eps=1e-12,                  # ε
        log_freq=50,                # GPU→CPU sync frequency for logging
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
            spectral_update_freq=spectral_update_freq,
            power_iterations=power_iterations,
            spec_c0=spec_c0,
            spec_r=spec_r,
            spec_tau_start=spec_tau_start,
            spec_tau_end=spec_tau_end,
            eps=eps,
        )
        super().__init__(params, defaults)
        self._global_step = 0
        self._log_freq = int(log_freq)
        self._last_log = {}

    def pop_last_log(self) -> dict:
        out = self._last_log
        self._last_log = {}
        return out

    # ─────────────────────────────────────────────────────────────────
    # Warm-started power iteration: returns σ_1(W) and ρ(W) = σ_1² / ||W||_F²
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _power_iter_sigma1_rho_warm(self, W, state, n_iter, eps):
        Wf = W.float()
        m, n = Wf.shape
        if m < 2 or n < 2:
            return 0.0, 0.0

        v = state.get("pi_v", None)
        if v is None or v.numel() != n:
            v = torch.randn(n, device=Wf.device, dtype=Wf.dtype)
        v = v / (v.norm() + eps)

        sigma1 = None
        for _ in range(max(1, int(n_iter))):
            u = Wf.mv(v)
            sigma1 = u.norm()
            u = u / (sigma1 + eps)
            v = Wf.t().mv(u)
            v = v / (v.norm() + eps)
        state["pi_v"] = v.detach()

        fro2 = Wf.pow(2).sum()
        conc = (sigma1 * sigma1 / (fro2 + eps)).clamp(0.0, 1.0)
        return float(sigma1.item()), float(conc.item())

    @torch.no_grad()
    def _tau_anneal(self, c, c0, r, tau_start, tau_end):
        """Eq.(thr_schedule):  τ(c) = τ_start + (τ_end - τ_start)/r · min(max(c - c_0, 0), r)."""
        if r <= 0:
            return float(tau_end)
        t = min(max(c - c0, 0), r)
        return float(tau_start + (tau_end - tau_start) * (t / float(r)))

    # ─────────────────────────────────────────────────────────────────
    # SpecFlag inner loop for a single matrix (Alg.2)
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _update_structural_flag(self, p, group):
        st = self.state[p]
        eps = float(group["eps"])

        # One-time state init
        if "spec_count" not in st:
            st["spec_count"] = 0
            st["mu_sigma1"] = torch.zeros((), device=p.device)
            st["nu_sigma1"] = torch.zeros((), device=p.device)
            st["mu_rho"]    = torch.zeros((), device=p.device)
            st["nu_rho"]    = torch.zeros((), device=p.device)
            st["specSens"]  = 0.0  # Python float — binary scalar 0/1

        if p.dim() != 2:
            st["specSens"] = 0.0
            return 0.0

        freq = max(1, int(group["spectral_update_freq"]))
        beta_eff = float(group["beta_utility"]) ** freq   # β_eff = β_u^f

        sigma1, rho = self._power_iter_sigma1_rho_warm(
            p.data, st, group["power_iterations"], eps=eps
        )
        sigma1_t = torch.tensor(sigma1, device=p.device)
        rho_t    = torch.tensor(rho,    device=p.device)

        # ── Snapshot previous stats (clone to avoid in-place aliasing) ──
        # 같은 텐서 참조를 in-place 변경하면 ν 업데이트의 (x - μ_{c-1})²가
        # 사실은 (x - μ_c)²가 되어버림 → clone()으로 격리.
        mu_s_prev = st["mu_sigma1"].clone()
        nu_s_prev = st["nu_sigma1"].clone()
        mu_r_prev = st["mu_rho"].clone()
        nu_r_prev = st["nu_rho"].clone()

        # ── Stabilized variance ν̃ = max(ν, ε)  (notation 표의 \widetilde ν) ──
        v_s_tilde = nu_s_prev.clamp(min=eps)
        v_r_tilde = nu_r_prev.clamp(min=eps)

        # Eq.(zscore):  z_x = (x - μ_{c-1}) / (√ν̃_{c-1} + ε)
        # 코드와 수도코드 정확히 일치하는 형태.
        z_sigma = (sigma1_t - mu_s_prev) / (v_s_tilde.sqrt() + eps)
        z_rho   = (rho_t    - mu_r_prev) / (v_r_tilde.sqrt() + eps)

        st["spec_count"] += 1
        c = int(st["spec_count"])

        c0        = int(group["spec_c0"])
        r         = int(group["spec_r"])
        tau_start = float(group["spec_tau_start"])
        tau_end   = float(group["spec_tau_end"])

        # Eq.(thr_schedule) + Eq.(sflag): burn-in, then annealed threshold
        if c < c0:
            s_flag = 0.0
        else:
            thr = self._tau_anneal(c, c0, r, tau_start, tau_end)
            unstable = torch.maximum(z_sigma, z_rho) > thr
            s_flag = 1.0 if bool(unstable.item()) else 0.0

        # Eq.(mu_update) + Eq.(m2_update):
        #   μ_c = β_eff · μ_{c-1} + (1-β_eff) · x
        #   ν_c = β_eff · ν_{c-1} + (1-β_eff) · (x - μ_{c-1})²
        # μ_*_prev는 위에서 clone() 했으므로 in-place 영향 받지 않음.
        one_minus = 1.0 - beta_eff
        st["mu_sigma1"].mul_(beta_eff).add_(sigma1_t, alpha=one_minus)
        st["nu_sigma1"].mul_(beta_eff).add_((sigma1_t - mu_s_prev).pow(2), alpha=one_minus)
        st["mu_rho"   ].mul_(beta_eff).add_(rho_t,    alpha=one_minus)
        st["nu_rho"   ].mul_(beta_eff).add_((rho_t    - mu_r_prev).pow(2), alpha=one_minus)

        # Alg.2 line: specSens[W] ← s_W (cache as scalar)
        st["specSens"] = s_flag
        return s_flag

    # ─────────────────────────────────────────────────────────────────
    # Main step (Alg.3)
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._global_step += 1
        do_log = (self._global_step % self._log_freq == 0)
        log_dict = {}

        # ── (A) SpecFlag check — group pause only on check steps (k mod f == 0)
        for gi, group in enumerate(self.param_groups):
            freq = max(1, int(group["spectral_update_freq"]))
            do_spec = (self._global_step % freq == 0)

            group_unstable = False
            if do_spec:
                flagged = 0
                for p in group["params"]:
                    if p.grad is None or p.dim() != 2:
                        continue
                    s_flag = self._update_structural_flag(p, group)
                    if s_flag >= 1.0:
                        group_unstable = True
                        flagged += 1
                if do_log:
                    log_dict[f"pico/g{gi}/spec_flagged_cnt"] = float(flagged)
                    log_dict[f"pico/g{gi}/pi_group_pause"]   = float(group_unstable)
            group["_noise_paused"] = bool(group_unstable) if do_spec else False

        # ── (B) Main update: GateU + perturbation-gated step (Alg.1 + Alg.3) ──
        for gi, group in enumerate(self.param_groups):
            paused = bool(group.get("_noise_paused", False))
            beta_u = float(group["beta_utility"])
            lr     = float(group["lr"])
            wd     = float(group["weight_decay"])
            sigma0 = float(group["sigma"])
            eps    = float(group["eps"])

            # Aggregations for periodic logging only
            agg_grad2 = agg_noise2 = agg_upd2 = 0.0
            agg_cnt = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state[p]

                # ── State init (one-time, persistent) ──────────────
                if "step" not in st:
                    st["step"] = 0
                    # avgU EMA (fp32, persistent in state)
                    st["avg_utility"] = torch.zeros_like(p.data, dtype=torch.float32)
                    # θ fp32 master copy (Alg.3 Require line; avoids per-step .float() churn)
                    st["theta_fp32"]  = p.data.detach().to(torch.float32).clone()
                if "specSens" not in st:
                    st["specSens"] = 0.0
                st["step"] += 1

                # ── GateU (Alg.1) ──────────────────────────────────
                # Transient fp32 view of grad; persistent fp32 master θ from state.
                g_fp32 = p.grad.detach().to(torch.float32)
                theta  = st["theta_fp32"]
                avg_u  = st["avg_utility"]

                # Eq.(avgU):  avgU ← β·avgU + (1-β)·(-g⊙θ)
                avg_u.mul_(beta_u).addcmul_(g_fp32, theta, value=-(1.0 - beta_u))

                # Eq.(biascorr)+(umax)+(gate) fused:
                #   u = σ(avgU_hat / u_max)
                #     = σ(avgU / ((1-β^k + ε) · u_max))
                # 추가 ε는 β=1.0 같은 edge case에서 division-by-zero 가드.
                # 일반 β<1, k≥1 에서는 (1-β^k) > 0 이므로 ε의 영향은 무시 가능.
                bias_corr = 1.0 - (beta_u ** st["step"])
                u_max     = avg_u.abs().max() + eps
                scale     = (bias_corr + eps) * u_max     # ε is a safety guard
                u_gate    = torch.sigmoid(avg_u / scale)

                # Eq.(protgrad):  g_tilde = g ⊙ (1 - u)
                g_tilde = g_fp32 * (1.0 - u_gate)

                # ── Structural flag s and group pause π ────────────
                s = float(st.get("specSens", 0.0))
                # Eq.(noise_gate):  m^(k) = (1-u)(1-s)(1-π)
                # m^(k) ≡ 0  iff  s=1 or π=1  → skip noise sampling.
                skip_noise = paused or (s >= 1.0)

                # Eq.(update):  θ ← (1 - ηλ)·θ - η·(g_tilde + ξ)
                if wd != 0.0:
                    theta.mul_(1.0 - wd * lr)

                if skip_noise:
                    # ξ = 0, no sampling
                    theta.add_(g_tilde, alpha=-lr)
                    if do_log:
                        upd_n = float(g_tilde.norm().item()) * lr
                        agg_upd2 += upd_n * upd_n
                else:
                    # Eq.(sigma_k):  σ_k = σ_0 · m^(k);  here s=π=0 so m = 1-u
                    # Eq.(noise)  :  ξ ~ N(0, σ_k² I)
                    noise = torch.randn_like(g_fp32) * (sigma0 * (1.0 - u_gate))
                    theta.add_(g_tilde + noise, alpha=-lr)
                    if do_log:
                        upd = (g_tilde + noise) * lr
                        agg_noise2 += float(noise.norm().item()) ** 2
                        agg_upd2   += float(upd.norm().item()) ** 2

                if do_log:
                    agg_grad2 += float(g_fp32.norm().item()) ** 2
                    agg_cnt += 1

                # Write back to param tensor (bf16/fp16 cast if needed)
                if p.data.dtype != torch.float32:
                    p.data.copy_(theta.to(dtype=p.data.dtype))
                else:
                    p.data.copy_(theta)

            if do_log and agg_cnt > 0:
                grad_norm  = math.sqrt(agg_grad2)
                noise_norm = math.sqrt(agg_noise2)
                upd_norm   = math.sqrt(agg_upd2)
                log_dict.update({
                    f"pico/g{gi}/pi_group_pause": float(paused),
                    f"pico/g{gi}/grad_norm":      grad_norm,
                    f"pico/g{gi}/noise_norm":     noise_norm,
                    f"pico/g{gi}/update_norm":    upd_norm,
                    f"pico/g{gi}/noise_to_grad":  noise_norm / (grad_norm + 1e-12),
                })

        if do_log:
            self._last_log = log_dict
        return loss
