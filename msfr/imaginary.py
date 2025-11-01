# 복소 범위에서 MSFR을 구현해서 성능을 비교해보자
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    """
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, trend=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_harmonics = n_harmonics
        self.trend = trend # 추세는 잠시 사용하지 않기로, 더 알아보고 싶음

        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=self.device))
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        self.cycle = Parameter(torch.empty(input_dim, device=self.device))
        nn.init.xavier_uniform_(self.weight) # xavier_uniform_: 입력, 출력 크기 기준으로 가중치 분산을 균형 있게 설정 -> 학습 안정성 향상
        nn.init.zeros_(self.bias) # bias를 0으로 초기화

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(-1) #x의 shape: (batch_size, input_dim, 1)
        harmonics = torch.arange(-self.n_harmonics-1, self.n_harmonics + 1, device=self.device).float()

        features = torch.exp(x * 2j * math.pi / self.cycle) * harmonics # 파이썬에서는 복소수 표현을 위해 접미사 j 사용
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias