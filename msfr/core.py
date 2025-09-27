#이 파일은 MSFR 라이브러리의 핵심 모델을 정의하는 파일임.
#pytoch 기반으로 여러 주기성을 가진 시계열 데이터를 사인 함수 특징으로 표현하고
#fit/predict 메서드를 통해 학습과 예측을 수행함.

import torch
import torch.nn as nn
import numpy as np

class MSFR(nn.Moudule):
    def __init__ (self, seasonal_periods, n_harmonics=3, trend="linear", reg_lambda=0.0, device = None):
        
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.n_harmonics = n_harmonics
        self.trend = trend
        self.reg_lambda = reg_lambda
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #일단 기본적인 틀은 이게 맞으니까 layers.py같은 이름의 파일에서 사인함수 계산 등을 하고 여기서 import해서 쓰면 될거 같음.
        #예를 들면 self.fourier_features = FourierFeatures(seasonal_periods, n_harmoics) 이런 식으로