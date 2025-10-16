import torch
import math
import torch.nn as nn
from torch.nn import functional as F # 뭔까 쓸거 같아서 일단 임포트
from torch.nn.parameter import Parameter

class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    - reg_lambda: 정규화 강도
    """
    
    def __init__ (self, input_dim, output_dim, n_harmonics=3, trend="linear", reg_lambda=0.0, device = None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_lambda = reg_lambda
        self.trend = trend
        
        self.weight = Parameter(torch.empty((output_dim, input_dim), device=self.device)) # 일단 다른 레이어랑 shape 맞추고, 계산에서 전치해서 행렬 곱 시 shape가 [batch, output_dim] 이 되도록 함
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        self.n_harmonics = n_harmonics # 얘도 파라미터로 할까

        # 이게 다변수 선형회귀처럼 하나의 x에 여러 feature를 넣을 수 있어야 하니까 input_dim이 필요
        # 근데 여기서 주기성을 지닌 feature가 있고, 추세가 있는 feature가 있는데, 이걸 어떻게 구분해야 하지

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input * 2 * math.pi) @ self.weight.T + self.bias # 프로토타입으로 가중치랑 바이어스만 부여 (행렬 곱 연산)
