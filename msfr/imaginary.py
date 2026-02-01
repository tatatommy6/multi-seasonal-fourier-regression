import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Optional

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스 (복소수 버전).

    - n_harmonics:주기별 세밀함 정도
    - init_cycle: 주기 초기값
    - trend: 계절성 외에 전체 추세 반영 방식
    """
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, trend="None", init_cycle=None, device=None):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.trend = trend

        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=device))
        self.bias = Parameter(torch.empty(output_dim, device=device))
        self.log_cycle  = Parameter(torch.empty(input_dim, device=device))
        self.reset_parameters(init_cycle)
        
    def reset_parameters(self, init_cycle : Optional[torch.Tensor]):
        nn.init.xavier_uniform_(self.weight) # 선형회귀의 kaiming_uniform_은 ReLU 가정이라 적합하지 읺다는 의견이 있어 변경
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.log_cycle, -2.0, 2.0)
        if init_cycle is not None:
            self.log_cycle.data = torch.log(torch.exp(init_cycle) - 1.0)

    @property
    def cycle(self) -> torch.Tensor:
        """
        MSFR 레이어의 주기 파라미터를 반환합니다.
        하지만 해당 파라미터를 직접 수정하지 마세요. 대신 init_cycle 인자를 사용하세요.
        """
        return F.softplus(self.log_cycle) + 1e-3

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        harmonics = torch.arange(1, self.n_harmonics + 1, device=input.device).float()  # (n_harmonics,)
        cycles = F.softplus(self.log_cycle) + 1e-3

        # 브로드캐스팅을 위해 차원 정렬
        x = input.unsqueeze(-1)                 # (batch_size, input_dim, 1)
        harmonics = harmonics.view(1, 1, -1)    # (1, 1, n_harmonics)
        cycle = cycles.view(1, -1, 1)           # (1, input_dim, 1)

        angle = x * (2 * math.pi) * harmonics / cycle  # (batch_size, input_dim, n_harmonics)
        features = torch.exp(angle)
        features = features.view(features.size(0), -1) # (batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        if self.trend != None: # 추세 항 추가 구현 (테스트 필요)
            if self.trend == "linear":
                trend_features = input
            elif self.trend == "quadratic":
                trend_features = input ** 2 # quadratic이 의미를 가질 수 있다는 것에 동의한다고 생각하여 다시 살림

            else:
                raise ValueError("trend must be one of [None, 'linear', 'quadratic']")

            features = torch.cat([features, trend_features], dim=-1)

        return features @ self.weight.T + self.bias