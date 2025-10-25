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
        harmonics = torch.arange(1, self.n_harmonics + 1, device=self.device).float()
        
        sin_terms = torch.sin(x * 2 * math.pi / self.cycle) * harmonics
        cos_terms = torch.cos(x * 2 * math.pi / self.cycle) * harmonics
        features = torch.cat([sin_terms, cos_terms], dim=-1) #torch.cat() : 여러개의 텐서를 하나로 연결하는 함수 (이때 텐서들의 차원은 다 같아야함)
        features = features.view(features.size(0), -1) #(batch_size, input_dim * 2 * n_harmonics) 형태로 flatten

        #TODO: 추세 항 추가 구현
        return features @ self.weight.T + self.bias
    
    # 실제 푸리에 급수와 식을 비교한 결과, 우리는 시그마로 더한 값에 전체 가중치를 행렬 곱 하고 있지만, 푸리에 급수는 각 사인/코사인항마다 개별 가중치를 곱하고 더하는 방식임.
    # 푸리에 급수의 방식을 그대로 따를 것인지, 아니면 지금 방식이 학습에 더 효율적이므로 유지할 것인지 고민 필요.
    # 정규화를 적용해서 모델 구조 안에 새로 손실함수를 제작해야 하는데, 이 부분은 레이어로써의 MSFR의 역할을 해침
    # 그래서 지금 방식을 적용하여 약간의 정규화를 준 후, 정규화 역할을 할 수 있는 손실함수를 새로 만드는 방향으로 진행하는 것이 좋아 보임.


    # 푸리에 급수의 일반항 말고, 계산해서 단순화 시킨 후 torch.arange에 스텝?을 파라미터로 주는 방식으로 개선하면 더 효율적일 수 있음.
    # 이러면 정규화도 쉽고, 계산량도 줄어들 것 같음.
    # 더 알아보고 개선하기로, 그리고 여전히 손실함수 새로 제작하는 것도 고려해야 함 
