import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    """

    # TODO: 주석이 너무 많아 복잡해보임 -> 정리 필요
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, grad_scale=1.0, trend=None, device=None):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.trend = trend # 추세는 잠시 사용하지 않기로, 더 알아보고 싶음

        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=device))
        self.bias = Parameter(torch.empty(output_dim, device=device))
        self.cycle = Parameter(torch.empty(input_dim, device=device))
        self.reset_parameters()
        
        # 파라미터들에 대한 그래디언트 후크 설정 -> 그레디언트 스케일링 -> 학습 반영 증가
        g = lambda g: torch.clamp(g * grad_scale, -1e6, 1e6)
        self.weight.register_hook(g)
        self.bias.register_hook(g)
        self.cycle.register_hook(g)

    def reset_parameters(self) -> None:
        # 선형회귀 초기화 방식에서 참고
        nn.init.xavier_uniform_(self.weight, gain=2.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cycle, 1.0, 10.0)  # 주기 초기값을 1~10 사이로 설정

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std # 정규화

        harmonics = torch.arange(1, self.n_harmonics + 1, device=input.device).float()  # (n_harmonics,)
        cycles = F.softplus(self.cycle) # 주기 양수화

        # 브로드캐스팅을 위해 차원 정렬
        x = x_norm.unsqueeze(-1)                # (batch_size, input_dim, 1)
        harmonics = harmonics.view(1, 1, -1)    # (1, 1, n_harmonics)
        cycle = cycles.view(1, -1, 1)           # (1, input_dim, 1)

        angle = x * (2 * math.pi) * harmonics / cycle  # (batch_size, input_dim, n_harmonics)
        sin_terms = torch.sin(angle)
        cos_terms = torch.cos(angle)
        features = torch.cat([sin_terms, cos_terms], dim=-1) #torch.cat() : 여러개의 텐서를 하나로 연결하는 함수 (이때 텐서들의 차원은 다 같아야함)
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias
