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
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, trend="linear", reg_lambda=0.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_harmonics = n_harmonics
        self.trend = trend
        self.reg_lambda = reg_lambda

        # weight 크기 = output_dim x (input_dim * (2*n_harmonics + trend_term))
        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=self.device))
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(-1)
        harmonics = torch.arange(1, self.n_harmonics + 1, device=self.device).float()
        sin_terms = torch.sin(x * harmonics * 2 * math.pi)
        cos_terms = torch.cos(x * harmonics * 2 * math.pi)
        features = torch.cat([sin_terms, cos_terms], dim=-1)
        features = features.view(features.size(0), -1)

