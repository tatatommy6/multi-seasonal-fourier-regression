import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    - reg_lambda: 정규화 강도
    """
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, trend=None, reg_lambda=0.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_harmonics = n_harmonics
        self.trend = trend # 추세는 잠시 사용하지 않기로, 더 알아보고 싶음
        self.reg_lambda = reg_lambda

        # weight 크기 = output_dim x (input_dim * (2*n_harmonics + trend_term))
        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=self.device)) #torch.empty()를 이용하여 텐서를 만들기만 하고 아직 채우진 않음
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        nn.init.xavier_uniform_(self.weight) # xavier_uniform_: 입력, 출력 크기 기준으로 가중치 분산을 균형 있게 설정 -> 학습 안정성 향상
        nn.init.zeros_(self.bias) # bias를 0으로 초기화

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(-1) #x의 shape: (batch_size, input_dim, 1)
        harmonics = torch.arange(1, self.n_harmonics + 1, device=self.device).float()
        sin_terms = torch.sin(x * harmonics * 2 * math.pi)
        cos_terms = torch.cos(x * harmonics * 2 * math.pi)
        features = torch.cat([sin_terms, cos_terms], dim=-1) #torch.cat() : 여러개의 텐서를 하나로 연결하는 함수 (이때 텐서들의 차원은 다 같아야함)
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias