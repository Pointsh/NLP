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
```Python
class EmoDataset(Dataset):
    def __init__(
        self,
        data,
        twt_tokenizer,
        roberta_tokenizer,
        label_encoder,
        max_length=256,
        mode=None,
    ):
        self._label_encoder = label_encoder
        self._twt_tokenizer = twt_tokenizer
        self._roberta_tokenizer = roberta_tokenizer
        self._max_length = max_length
        self._mode = mode
        self._dataset = self._init_dataset(data)

    def _init_dataset(self, data): #dataset 정리
        data["Utterance"] = data["Utterance"].map(self._preprocess)

        if self._mode == "train":
            data["Target"] = data["Target"].map(self._label_encoder.encode)
            data = data.loc[:, ["Utterance", "Target"]]
        else:
            data = data.loc[:, "Utterance"]
            data = data.to_frame()
        return data

    def __len__(self): #길이값 추출
        return len(self._dataset)
```
roberta_tokenizer 사용하여 토근화
이 tokenizer는 공백을 토큰의 일부(문장과 비슷하게)처럼 처리하도록 훈련되어있으므로 단어는 문장의 시작부분에 상관없이 다르게 encodig됩니다.
```Python 
    def __getitem__(self, idx): 
        text = self._dataset.loc[idx, "Utterance"]
        inputs = self._roberta_tokenizer(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0] #inputids 뽑아내기
        attention_mask = inputs["attention_mask"][0] #attention_mask 뽑아내기

        if self._mode == "train":
            y = self._dataset.loc[idx, "Target"]
            return input_ids, attention_mask, y
        else:
            return input_ids, attention_mask

    def _preprocess(self, sentence):
        twt_tokens = self._twt_tokenizer.tokenize(sentence) #TweetTokenizer(이모티콘 인식 가능) 진행 [대화 데이터셋]
        twt_sentence = self._tokens_to_sentence(twt_tokens) 
        return twt_sentence

    def _tokens_to_sentence(self, tokens): #TweetTokenizer로 나뉘어져 있는 단어들을 이어붙여 전처리 진행
        sentence = " ".join(tokens)
        marks = re.findall(r"\s\W\s*", sentence)
        for mark in marks:
            if mark.strip() in ["'", "’"]:
                sentence = sentence.replace(mark, mark.strip())
            else:
                sentence = sentence.replace(mark, mark.lstrip())
        return sentence
```
data 부르기
```Python
train_csv = pd.read_csv(TRAIN_CSV)

df_train, df_val = train_test_split(
    train_csv, test_size=0.1, shuffle=True, random_state=32
)
df_train.head()
```
![캡처](https://user-images.githubusercontent.com/44185037/210360577-908f9fdb-93e3-440c-927e-ca357dc48c8e.JPG)

적용
```Python
label_encoder = LabelEncoder()
roberta_tokenizer = RobertaTokenizerFast.from_pretrained(ARGS["model"], truncation=True)
twt_tokenizer = nltk.tokenize.TweetTokenizer(
    preserve_case=False, strip_handles=True, reduce_len=True
)

train_set = EmoDataset(
    df_train.reset_index(drop=True),
    twt_tokenizer,
    roberta_tokenizer,
    label_encoder,
    max_length=ARGS["max_len"],
    mode="train",
)
val_set = EmoDataset(
    df_val.reset_index(drop=True),
    twt_tokenizer,
    roberta_tokenizer,
    label_encoder,
    max_length=ARGS["max_len"],
    mode="train",
)

train_dataloader = DataLoader(train_set, batch_size=ARGS["batch_size"])
val_dataloader = DataLoader(val_set, batch_size=ARGS["batch_size"])
```
Model 실행 def
```Python
def evaluate(model, criterion, val_loader, device, mode=None):
    model.eval()

    val_loss = list()
    model_preds = list()
    true_labels = list()

    with torch.no_grad():
        for input_ids, attention_mask, label in val_loader:
            label = label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output.logits, label.long())
            val_loss.append(batch_loss.item())

            if mode != "train":
                model_preds += output.logits.argmax(1).detach().cpu().numpy().tolist()
                true_labels += label.detach().cpu().numpy().tolist()

        if mode != "train":
            val_acc = accuracy_score(true_labels, model_preds)
            val_f1 = f1_score(true_labels, model_preds, average="macro")
            return val_acc, val_f1

        return val_loss


def train(model, optimizer, scheduler, criterion, train_loader, val_loader, device):
    torch.cuda.empty_cache()
    gc.collect()

    epoch_progress = trange(1, ARGS["epochs"] + 1)
    early_stopper = EarlyStoppingCallback(patience=ARGS["patience"])
    criterion.to(device)

    grad_step = ARGS["grad_step"]
    model_path = ARGS["model_path"]
    model.to(device)
    model.zero_grad()

    for epoch in epoch_progress:

        model.train()
        train_loss = list()
        for batch_id, data in enumerate(train_loader, start=1):

            input_ids, attention_mask, train_label = data
            train_label = train_label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output.logits, train_label.long())
            train_loss.append(batch_loss.item())

            batch_loss /= grad_step
            batch_loss.backward()

            if batch_id % grad_step == 0:
                optimizer.step()
                model.zero_grad()

        val_loss = evaluate(model, criterion, val_loader, device, mode="train")
        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        tqdm.write(
            f"Epoch {epoch}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
        )

        if early_stopper.should_stop(val_loss, model, model_path):
            tqdm.write(
                f"\n\n -- EarlyStoppingCallback: [Epoch: {epoch - ARGS['patience']}]"
            )
            tqdm.write(f"Model saved at '{model_path}'.")
            break

        scheduler.step()

    model.load_state_dict(torch.load(model_path))
    return model

model = RobertaForSequenceClassification.from_pretrained(
    ARGS["model"], num_labels=label_encoder.num_classes
)
optimizer = AdamW(
    model.parameters(), lr=ARGS["lr"], weight_decay=0.01, correct_bias=False
)
scheduler = LambdaLR(
    optimizer, lr_lambda=lambda epoch: 0.9**epoch
)
criterion = CrossEntropyLoss()
```
Train진행
```Python
best_model = train(
    model, optimizer, scheduler, criterion, train_dataloader, val_dataloader, DEVICE
)

torch.cuda.empty_cache()
gc.collect()

val_acc, val_f1 = evaluate(best_model, criterion, val_dataloader, DEVICE)
print(f"Accuracy: {val_acc:.5f}")
print(f"F1-macro: {val_f1:.5f}")
```
Accuracy : 0.73678
F1-macro : 0.65121

해당 모델을 Test에 적용합니다.
```Python
df_test = pd.read_csv(TEST_CSV)

test_set = EmoDataset(
    df_test.reset_index(drop=True),
    twt_tokenizer,
    roberta_tokenizer,
    label_encoder,
    max_length=ARGS["max_len"],
)
test_dataloader = DataLoader(test_set, batch_size=ARGS["batch_size"], shuffle=False)

def predict(model, test_loader, device):
    model.eval()
    model_preds = list()

    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)
            model_preds += output.logits.argmax(1).detach().cpu().numpy().tolist()
        return model_preds
```
실행합니다.
```Python
preds = predict(best_model, test_dataloader, DEVICE)
```
이후 해당 값들을 다시 Labeldecode를 통해 변경하여 값들을 넣습니다.
```Python
df_test["Target"] = preds
df_test["Target"] = df_test["Target"].map(label_encoder.decode)
submission = df_test.loc[:, ["ID", "Target"]]
```
정리된 값 제출 결과 0.55012
이미 종료된 프로젝트지만 약 9등정도의 값을 도출해 냈습니다.