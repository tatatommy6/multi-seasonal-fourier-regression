import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
torch.nn.Linear
class MSFR(nn.Module):
    """
    Multi-Seasonal Fourier Regression (MSFR) 레이어 클래스.

    - n_harmonics:주기별 세밀함 정도
    - trend: 계절성 외에 전체 추세 반영 방식
    """
    
    def __init__(self, input_dim, output_dim, n_harmonics=3, trend=None, device=None):
        super().__init__()
        # self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.n_harmonics = n_harmonics
        self.trend = trend # 추세는 잠시 사용하지 않기로, 더 알아보고 싶음

        trend_dim = input_dim if trend in ["linear", "quadratic"] else 0
        total_features = input_dim * (2 * n_harmonics) + trend_dim

        self.weight = Parameter(torch.empty((output_dim, total_features), device=self.device))
        self.bias = Parameter(torch.empty(output_dim, device=self.device))
        self.cycle = Parameter(torch.empty(input_dim, device=self.device)) # 주기를 파라미터화
        # nn.init.xavier_uniform_(self.weight) # xavier_uniform_: 입력, 출력 크기 기준으로 가중치 분산을 균형 있게 설정 -> 학습 안정성 향상
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 선형회귀 초기화 방식
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cycle, 1.0, 10.0)  # 주기 초기값을 1~10 사이로 설정

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(-1) # (batch_size, input_dim, 1)
        harmonics = torch.arange(1, self.n_harmonics + 1, device=self.device).float()  # (n_harmonics,)
        # 브로드캐스팅을 위해 차원 정렬
        harmonics = harmonics.view(1, 1, -1)  # (1, 1, n_harmonics)
        cycle = self.cycle.view(1, -1, 1)     # (1, input_dim, 1)

        angle = x * (2 * math.pi) * harmonics / cycle  # (batch_size, input_dim, n_harmonics)
        sin_terms = torch.sin(angle)
        cos_terms = torch.cos(angle)
        features = torch.cat([sin_terms, cos_terms], dim=-1) #torch.cat() : 여러개의 텐서를 하나로 연결하는 함수 (이때 텐서들의 차원은 다 같아야함)
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias
    