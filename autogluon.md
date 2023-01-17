# 기존 Dacon 감정발화를 autogluon으로 구현한 부분을 설명하겠습니다.

```python
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
!pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

!pip install autogluon

```
train data와 test data를 불러오고 torch, autogluon을 설치해주었습니다.

```python
from autogluon.multimodal import MultiModalPredictor #autogluon의 multimodal import MultiModalPredictor

sample_submission = pd.read_csv('sample_submission.csv')

train['Dialogue_ID'] = train['Dialogue_ID'].astype('category') # Dialogue_ID를 범주형으로 변환
test['Dialogue_ID'] = test['Dialogue_ID'].astype('category')

ag_train = train.drop('ID', axis=1) #필요없는 ID부분 제거
ag_test = test.drop('ID', axis=1)

ag_train['num_len']=ag_train.Utterance.apply(lambda x:len(x)) #num len의 길이를 알기위해 lambda함수 구현

ag_train.head(10)

```
![image](https://user-images.githubusercontent.com/44185037/210969194-4c618d1e-4888-4d4b-a5ea-a227d3ce10a5.png)

numlen이 잘 들어간 것을 확인할 수 있습니다.
여기서 1글자 들의 경우 감정분석과 전혀 상관없는 부분이라 생각하여 확인 후 제거작업 진행하였습니다.

```python
ag_train[(ag_train.num_len==1)]
```
![image](https://user-images.githubusercontent.com/44185037/210969461-b9d0ce15-13c9-4129-bcc2-78b0abaaac9b.png)

```python
ag_train.drop(ag_train[ag_train.num_len==1].index,inplace=True)
```
이후 2글자이나 Target이 'neutral'인 경우를 확인 후 제거하였습니다.
```python
ag_train[(ag_train.num_len==2)&(ag_train.Target=='neutral')]
```
![image](https://user-images.githubusercontent.com/44185037/210969638-d8c56bd4-ecbd-42d1-acd4-f2ae4adc6fc7.png)

```python
ag_train.drop(ag_train[(ag_train.num_len==2)&(ag_train.Target!='neutral')].index,inplace=True)

ag_train.drop('num_len',axis=1,inplace=True) #사용했던 num_len열 제거
```
autogluon의 hyperparameters 설정
```python
hyperparameters = {
    "model.names": ["hf_text", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
    "model.hf_text.checkpoint_name":'tae898/emoberta-base'
    }
```
텍스트 데이터용 huggingface 변환기 'hf_txt'사용, 범주데이터용 'categorical_mlp'사용, 숫자데이터용 'numerical_mlp'사용, text와 범주, 숫자 모두를 사용하기에 'fusion_mlp'사용
huggingface의 model을 불러오는 model.hf_text.checkpoint_name 사용
자세한내용 https://auto.gluon.ai/0.4.1/tutorials/text_prediction/automm.html

해당 parameter들과 이외의 parameter 설명은 아래와 같습니다.

* "hf_text": the pretrained text models from Huggingface.

* "timm_image": the pretrained image models from TIMM.

* "clip": the pretrained CLIP models.

* "categorical_mlp": MLP for categorical data.

* "numerical_mlp": MLP for numerical data.

* "categorical_transformer": FT-Transformer for categorical data.

* "numerical_transformer": FT-Transformer for numerical data.

* "fusion_mlp": MLP-based fusion for features from multiple backbones.

* "fusion_transformer": transformer-based fusion for features from multiple backbones.

```python
predictor = MultiModalPredictor(label='Target', problem_type='multiclass', eval_metric='log_loss')
predictor.fit(ag_train, hyperparameters=hyperparameters)
```
problem_type으로는 아래 4가지 중 multicalss를 사용하였습니다.

- 'binary': Binary classification
- 'multiclass': Multi-class classification
- 'regression': Regression
- 'classification': Classification problems include 'binary' and 'multiclass' classification.

이후 MultiModalPredictor를 이용하여 fit을 해줍니다.

macro f1 score를 사용하고 점수를 확인하였는데 낮은 점수를 기록하여 log_loss가 오버피팅에 더 낫다는 것이 기억이 나 해당 log_loss로 변경하니 좀 더 좋은 점수를 기록할 수 있었습니다.

학습이 전부 진행된 후
```python
predictions = predictor.predict(ag_test)
sample_submission['Target'] = predictions
sample_submission['Target']
```
결과를 저장하고 출력합니다.

![image](https://user-images.githubusercontent.com/44185037/211004660-f4495981-53de-4706-ac18-14b4435406b0.png)

```python
sample_submission.to_csv('submission_numlen_del_sep4.csv',index=False)
```
처음 Dacon 실제 공모전 기간에 참여해본 거라 많은 시간을 투자하며 hyperparameter등을 조정하였으나 이후 chatbot project가 있어서 마무리하였습니다..

제출하였을 땐 최종점수 0.524로 전체 500명중 26등이었으나
이후 마감일날 까지 확인을 하지않았더니 무슨이유에선지 점수가 0.49로 줄어들어 최종 39등으로 상위 16%로 마무리하였습니다.
![image](https://user-images.githubusercontent.com/44185037/210910892-c1be0398-8e39-4846-b2e4-6b0da1be5d4b.png)

**Auto ML이 아닌 RobertaForSequenceClassification로 구현한 모델은 해당 폴더의 nlp.md에 기술해놓았습니다.**


