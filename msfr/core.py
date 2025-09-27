#이 파일은 MSFR 라이브러리의 핵심 모델을 정의하는 파일임.(아님 모델이 아니라 딥러닝 레이어임)
#pytoch 기반으로 여러 주기성을 가진 시계열 데이터를 사인 함수 특징으로 표현하고
#fit/predict 메서드를 통해 학습과 예측을 수행함.(아님 forward 메서드임)

import torch
import torch.nn as nn
from torch.nn import functional as F # 뭔까 쓸거 같아서 일단 임포트
import numpy as np

class MSFR(nn.Module):
    def __init__ (self, seasonal_periods, n_harmonics=3, trend="linear", reg_lambda=0.0, device = None):
        
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.n_harmonics = n_harmonics
        self.trend = trend
        self.reg_lambda = reg_lambda
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #일단 기본적인 틀은 이게 맞으니까 layers.py같은 이름의 파일에서 사인함수 계산 등을 하고 여기서 import해서 쓰면 될거 같음.
        #   - layers.py도 좋은데 약간 계산적인 함수들만 넣는 파일을 만들고 여기서 불러와서 forward 구성하는게 더 나을수도?
        #예를 들면 self.fourier_features = FourierFeatures(seasonal_periods, n_harmoics) 이런 식으로

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1,1) # 임시값
        # return F.linear(input, self.weight, self.bias) 선형회귀 코드에서 가져옴