import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDistillationLoss(nn.Module):
    """
    KL Divergence + Huber(Smooth L1) 결합 Distillation Loss
    """
    def __init__(
        self,
        alpha_kl=1.0,        # KL Loss 가중치
        alpha_huber=1.0,     # Huber Loss 가중치
        huber_delta=1.0,     # Huber Loss의 delta (beta)
        temperature=1.0      # KL 계산 시 softmax 온도
    ):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.alpha_huber = alpha_huber
        self.huber_delta = huber_delta
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits=None, mask=None):
        """
        Args:
            student_logits: (B, seq_len, vocab_size)
            teacher_logits: (B, seq_len, vocab_size)
            mask: (B, seq_len) -> pad 위치를 0, 실제 토큰 위치를 1로 두면 무시 가능
        """
        if teacher_logits is None:
            # teacher 없음 -> loss=0
            return torch.tensor(0.0, device=student_logits.device)

        # (1) KLDiv Loss
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs     = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_per_token = F.kl_div(
            s_log_probs, 
            t_probs, 
            reduction='none'
        ).sum(dim=-1)  # vocab 차원 합 -> (B, seq_len)

        # 마스킹
        if mask is not None:
            kl_per_token = kl_per_token * mask

        kl_loss = kl_per_token.mean() * (self.temperature**2)

        # (2) Huber Loss
        # 여기서는 "log_probs(student) - log_probs(teacher)" 에 대한 Huber
        s_log = F.log_softmax(student_logits, dim=-1)
        t_log = F.log_softmax(teacher_logits, dim=-1)
        diff = s_log - t_log   # (B, seq_len, vocab_size)

        # smooth_l1_loss(reduction='none') -> (B, seq_len, vocab_size)
        huber_per_token = F.smooth_l1_loss(
            diff,
            torch.zeros_like(diff),
            beta=self.huber_delta,
            reduction='none'
        )
        # vocab 평균 후 seq_len 마스킹
        huber_per_token = huber_per_token.mean(dim=-1)  # (B, seq_len)
        if mask is not None:
            huber_per_token = huber_per_token * mask
        huber_loss = huber_per_token.mean()

        # (3) 최종 Loss 합산
        total_loss = self.alpha_kl * kl_loss + self.alpha_huber * huber_loss
        return total_loss
