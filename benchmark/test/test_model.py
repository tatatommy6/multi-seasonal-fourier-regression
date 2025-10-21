import torch
from msfr import MSFR
import torch.nn as nn
import numpy as np

# 여기에 프로토타입 사용해서 작동 확인 할 예정

class TestModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.msfr = MSFR(input_dim, output_dim)

    def forward(self, x):
        return self.msfr(x)