# Positional Encoding

* 이번에는 저번 논문 리뷰에서 다루었던 **Positional Encoding**을 pytorch를 통해 다뤄보겠습니다.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        """
        sin, cos encoding 구현
        
        parameter
        - d_model : model의 차원
        - max_len : 최대 seaquence 길이
        - device : cuda or cpu
        """
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # input matrix(자연어 처리에선 임베딩 벡터)와 같은 size의 tensor 생성
        # 즉, (max_len, d_model) size
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 인코딩의 그래디언트는 필요없습니다. 
        
        # 위치 indexing용 벡터
        # pos는 max_len의 index를 의미합니다.
        pos = torch.arange(0, max_len, device =device)
        # 1D : (max_len, ) size -> 2D : (max_len, 1) size -> word의 위치를 반영하기 위해
        
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32변환
        
        # i는 d_model의 index를 의미합니다. _2i : (d_model, ) size
        # 즉, embedding size가 512일 때, i = [0,512]
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        # (max_len, 1) / (d_model/2 ) -> (max_len, d_model/2)
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        
    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        # batch_size = 128, seq_len = 30
        batch_size, seq_len = x.size() 
        
        # [seq_len = 30, d_model = 512]
        # [128, 30, 512]의 size를 가지는 token embedding에 더해질 것입니다. 
        # 
        return self.encoding[:seq_len, :]
```