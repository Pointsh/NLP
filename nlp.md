Dacon 발화자의 감정인식 AI 경진대회
=================================

```python
import nltk
nltk.download("punkt")

import os
import re
import gc
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, RobertaTokenizerFast, RobertaForSequenceClassification
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

```
모델의 학습정보를 미리 선언합니다.
huggingface의 tae898의 emoberta-large를 활용하였습니다.

```python
ARGS = {
    "model": "tae898/emoberta-large",
    "model_path": 'emos_model.pth',
    "batch_size": 16,
    "grad_step": 16,
    "epochs": 10,
    "max_len": 128,
    "lr": 2e-7,
    "patience": 2, 
}
```
EarlyStoppingCallback Class 생성하여 계속하여 개선되지 않는다면 조기종료합니다.
```python
class EarlyStoppingCallback(object):
    def __init__(self, patience=2):
        self._min_eval_loss = np.inf
        self._patience = patience
        self.__counter = 0

    def should_stop(self, eval_loss, model, save_path):
        if eval_loss < self._min_eval_loss:
            self._min_eval_loss = eval_loss
            self.__counter = 0
            torch.save(model.state_dict(), save_path)
        elif eval_loss > self._min_eval_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
```
![github발췌](https://user-images.githubusercontent.com/44185037/210329829-b39b504c-a714-4bab-830c-5068ae61808b.JPG)

tae898의 github에 있는 source중 label부분과 동일한 레이블을 가지도록 조정했습니다,

```python
class LabelEncoder(object):
    def __init__(self):
        self._targets = [
            "neutral",
            "joy",
            "surprise",
            "anger",
            "sadness",
            "disgust",
            "fear",
        ]
        self.num_classes = len(self._targets)

    def encode(self, label):
        return self._targets.index(label)

    def decode(self, label):
        return self._targets[label]
```


