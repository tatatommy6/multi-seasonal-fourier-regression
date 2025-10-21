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
    
    def __init__ (self, input_dim, output_dim, n_harmonics=3, trend="linear", reg_lambda=0.0, device = None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_lambda = reg_lambda
        self.trend = trend
        
        self.weight = Parameter(torch.empty((output_dim, input_dim), device=self.device))
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        self.season = Parameter(torch.empty(input_dim, device=self.device))
        self.n_harmonics = n_harmonics # 일단 유지하고 필요하다면 파라미터화

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = self.weight.T # 진폭
        b = self.season / (2 * math.pi) # 주기 함수 계수
        # x축 평행 이동을 위한 c 변수가 과연 필요할까
        d = self.bias # y축 평행 이동

        # input sixe : [batch, input_dim]
        # a size : [input_dim, output_dim]
        # return value size: [batch, output_dim]

        return (torch.sin(input * b) @ a) + d
