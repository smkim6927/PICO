import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

class NoiseInjecter(nn.Module):
    def __init__(self, model, ema_decay=0.99, noise_scale=0.01, adjust_noise_interval=100, output_dir="./plots"):
        super().__init__()
        self.model = model
        self.ema_decay = ema_decay
        self.noise_scale = noise_scale
        self.adjust_noise_interval = adjust_noise_interval
        self.step = 0

        # EMA 텐서를 버퍼로 등록 (버퍼 이름에서 '.'을 '_'로 대체)
        for name, param in model.named_parameters():
            safe_name = f"ema_{name.replace('.', '_')}"
            self.register_buffer(safe_name, torch.zeros_like(param))

        # 추적용 리스트
        self.noise_scale_history = []
        self.ema_history = []

        # 그래프 저장 경로
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def forward(self, **kwargs):
        if self.training:
            self.inject_noise()
        return self.model(**kwargs)

    def inject_noise(self):
        for name, param in self.model.named_parameters():
            # 노이즈 생성
            noise = torch.randn_like(param) * self.noise_scale

            # 노이즈 주입
            param.data.add_(noise)

            # EMA 업데이트 (버퍼 이름에서 '.'을 '_'로 대체)
            safe_name = f"ema_{name.replace('.', '_')}"
            ema = getattr(self, safe_name)
            ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * noise.abs())

        # 현재 노이즈 스케일과 EMA 평균 기록
        avg_ema_value = self.get_average_ema()
        self.noise_scale_history.append(self.noise_scale)
        self.ema_history.append(avg_ema_value)

    def adjust_noise_scale(self):
        if self.step % self.adjust_noise_interval == 0:  # 예시: 100 스텝마다
            for name, param in self.model.named_parameters():
                # EMA 값에 따라 노이즈 스케일 조정 (버퍼 이름에서 '.'을 '_'로 대체)
                safe_name = f"ema_{name.replace('.', '_')}"
                ema = getattr(self, safe_name)
                param_noise_scale = self.noise_scale * (1 + ema.mean().item())
                param.data.add_(torch.randn_like(param) * param_noise_scale)

    def get_average_ema(self):
        """
        모든 EMA 버퍼의 평균값을 계산합니다.
        """
        ema_values = []
        for name, buffer in self.named_buffers():
            if name.startswith("ema_"):
                ema_values.append(buffer.mean().item())
        return sum(ema_values) / len(ema_values) if ema_values else 0.0

    def save_tracking_plots(self):
        """
        노이즈 스케일과 EMA 값 변화를 플롯으로 저장합니다.
        
        - 그래프 색상: 밝은 파란색 (`lightblue`)
        - 주요 포인트에 `+` 마커 표시.
        
        저장 경로:
          - noise_scale_plot.png
          - ema_value_plot.png
        """
        
        # 노이즈 스케일 플롯 저장
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(len(self.noise_scale_history)),
            self.noise_scale_history,
            label="Noise Scale",
            color="lightblue",
            marker="+",
            markersize=8,
            linestyle="-"
        )
        plt.xlabel("Steps")
        plt.ylabel("Noise Scale")
        plt.title("Noise Scale Over Time")
        plt.legend()
        
        noise_plot_path = os.path.join(self.output_dir,
         f"noise_scale_plot_{len(self.noise_scale_history)}.png")
        plt.savefig(noise_plot_path)
        
        plt.close()

        # EMA 값 플롯 저장
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(len(self.ema_history)),
            self.ema_history,
            label="Average EMA Value",
            color="lightblue",
            marker="+",
            markersize=8,
            linestyle="-"
        )
        plt.xlabel("Steps")
        plt.ylabel("EMA Value")
        plt.title("EMA Value Over Time")
        plt.legend()
        
        ema_plot_path = os.path.join(self.output_dir, "ema_value_plot.png")
        
        plt.savefig(ema_plot_path)

