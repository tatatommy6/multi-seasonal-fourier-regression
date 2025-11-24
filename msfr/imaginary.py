import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스 (복소수 버전).

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    """
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, init_cycle=None, trend=None, device=None):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.trend = trend

        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=device))
        self.bias = Parameter(torch.empty(output_dim, device=device))
        self.cycle = Parameter(torch.empty(input_dim, device=device))
        self.reset_parameters(init_cycle) # type: ignore
        
    def reset_parameters(self, init_cycle : torch.Tensor): # 선형회귀 초기화 + 주기값 초기화
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cycle, 0.5, 10.0)
        if init_cycle is not None:
            self.cycle.data = init_cycle

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        harmonics = torch.arange(1, self.n_harmonics + 1, device=input.device).float()  # (n_harmonics,)
        cycles = F.softplus(self.cycle) + 1e-3

        # 브로드캐스팅을 위해 차원 정렬
        x = input.unsqueeze(-1)                # (batch_size, input_dim, 1)
        harmonics = harmonics.view(1, 1, -1)    # (1, 1, n_harmonics)
        cycle = cycles.view(1, -1, 1)           # (1, input_dim, 1)

        angle = x * (2 * math.pi) * harmonics / cycle  # (batch_size, input_dim, n_harmonics)
        features = torch.exp(angle)
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias