import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics: 주기별 세밀함 정도
    - init_cycle: 주기 초기값
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
        sin_terms = torch.sin(angle)
        cos_terms = torch.cos(angle)
        features = torch.cat([sin_terms, cos_terms], dim = -1) # (batch_size, input_dim, 2 * n_harmonics)
        features = features.view(features.size(0), -1) # flatten

        if self.trend != None: # 추세 항 추가 구현 (테스트 필요)
            if self.trend == "linear":
                trend_features = input
            elif self.trend == "quadratic":
                trend_features = input ** 2 # quadratic이 의미를 가질 수 있다는 것에 동의한다고 생각하여 다시 살림
            
            # 그리고 다시 생각해봤는데 그래프가 완전 겹치면 모델이 잘못 예측하고 있는거임
            # '예측' 이잖아. 예측이니까 미래를 그린 곡선이라는 거니까 과거와 완전히 겹치는 곡선이 아니여야 함
            # - 주기성이 있는 데이터니까 과거와 비슷한 모양이여야 맞는 거임 완전히 겹칠 필요는 없고 그냥 비슷한 모양이면 됨
            #   -> 음 ㅇㅋ 그니까 비슷한 모양이 나오게 하는데 완전히 겹치지는 않고 조금 앞으로 가있는 모양이 나오면 가장 좋은거지
            #       - 아니 앞으로 가면 안되지 현재의 값을 예측할 수 있어야 미래의 값도 예측할 수 있는거니까
            else:
                raise ValueError("trend must be one of [None, 'linear', 'quadratic']")

            features = torch.cat([features, trend_features], dim=-1)

        return features @ self.weight.T + self.bias
