BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding
===============================================================================

## 해당 페이지의 구성은 논문을 쭉 읽어나가며 공부하고 정리한 포스트입니다.


- BERT : Bidirectional Encoder Representations form Transformer
해당 논문의 제목입니다. 해당 논문은 “Attention is all you need(Vaswani et al., 2017)”에서 소개한 **Transformer** 구조를 활용한 언어표현에 관한 논문이라고합니다.
Transformer에 대한 구조는 아래와 같습니다.

![트랜스포머 구조](https://user-images.githubusercontent.com/44185037/210715269-a08e3167-1f4d-42b1-93e8-1a3eca136a0e.JPG)

- 트랜스포머는 Attention 기법만 사용하기때문에 RNN,CNN은 전혀 사용하지않습니다.
* 그러나 기존의 **seq2seq**의 인코더-디코더 구조는 유지합니다.
* 인코더에서 입력 시퀀스를 입력받고, 디코더에서 출력 시퀀스를 출력합니다.

* 구조가 위와 같기때문에 문장 안에 포함된 각각의 단어들의 순서에 대한 정보를 주기 어렵습니다.
* 문장내 각각 단어의 순서에 대한 정보를 알려주기 위해 **Positional encoding**을 사용합니다
* 이러한 아키텍처는 **BERT**와 같은 향상된 네트워크에서도 채택되었습니다.
* Attention 과정 한번만 사용하는 것이 아니라 여러 Layer를 거쳐서 반복하도록 구현되어있습니다.

여기서 **Positional encoding**에 대해 짚고 넘어가겠습니다.

![positional](https://user-images.githubusercontent.com/44185037/210715383-ee5b1199-1b95-431a-843e-aa3469e22466.JPG)

## Positional encoding이란
- Transformer는 입력된 데이터를 한번에 병렬로 처리해서 속도가 빠르다는 장점이 있습니다.
하지만 RNN과 LSTM과 다르게 Transformer는 입력 순서가 단어 순서에 대한 정보를 보장하지 않으니 Transformer의 경우
squence가 한번에 병렬로 입력되기때문에 단어 순서에 대한 정보가 사라집니다. 따라서 단어 위치 정보를 별도로 넣어줘야합니다.
단어의 위치 정보는 왜 중요한지 이문제를 Positional encoding으로 어떻게 해결하는지 알아보겠습니다.

1. 'Although i did **not** get 95 in last TOEFL, I could get in the Ph.D program
2. 'Although i did get 95 in last TOEFL, I could **not** get in the Ph.D program

- 위의 두 문장을 해석해보면 1번 문장의 경우 '지난 토플시험에서 95점을 못 받았지만, 박사과정에 입학할 수 있었다'가 되고
2번 문장의 경우 '지난 토플 시험에서 95점을 받았지만, 박사과정에 입학하지 못했다'가 됩니다.  not의 위치로 인해
두 문장의 뜻이 완전히 달라졌습니다. 이와 같이 문장 내의 정확한 단어 위치를 알 수 없다면 문장의 뜻이 완전히 달라지는 문제가 발생합니다.

![캡처](https://user-images.githubusercontent.com/44185037/210727770-d4a90292-966b-4e43-a066-bc123ded3e77.JPG)

* 위의 그림은 하나의 단어를 vector화 시킨 모습입니다. 그 옆에 그림과 같이 Positional Encoding을 통해 얻은 위치정보를 더해주야합니다. 이때 지켜야 될 규칙 두가지가 있습니다.

1. 모든 위치값은 sequence의 길이나 input에 관계없이 동일한 식별자를 가져야합니다. 따라서 sequence가 변경되더라도 위치 embedding은 동일하게 유지됩니다.

![캡처](https://user-images.githubusercontent.com/44185037/210728014-67a5159d-0b27-493d-9059-183dfdeef26c.JPG)

2. 모든 위치값은 너무 크면 안됩니다. 위치값이 너무 커져버리면 단어간의 상관관계 및 의미를 유추할 수 있는 의미정보 값이 상대적으로 작아지게되고, Attention layer에서 제대로 학습 및 훈련이 되지 않을 수 있습니다.

![캡처](https://user-images.githubusercontent.com/44185037/210728485-ec6d4641-a82b-4a51-a797-352860849126.JPG)


- 위의 두 가지 규칙을 지키면서 vector를 부과하는 방법에는 sine & cosine 함수가 있습니다.

* 해당 sine & cosine 함수가 positional encoding에서 사용한 이유는 총 3가지가 있습니다.

1. 의미정보가 변질되지 않도록 위치 vector값이 너무 크면 안된다.
    - sine & cosine 함수는 -1 ~ 1 사이를 반복하는 주기함수입니다. 즉 1을 초과하지않고 -1 미만으로 떨어지지 않으므로 값이 너무 커지지 않는 조건을 만족시킵니다.

2. sine & cosine 함수 외에도 일정 구간 내에 있는 함수로는 Sigmoid 함수가 있습니다. 해당 함수를 채택하지않고 sine & cosine 함수를 선택한 이유는 **주기함수**이기 떄문입니다.

![캡처](https://user-images.githubusercontent.com/44185037/210729740-45c6d201-407f-41e0-bb75-bde349a6bc15.JPG)

- 위의 그림과 같이 Sigmoid 함수의 경우, 긴 문장의 sequence가 주어진 경우, 위치 vetor 값의 차가 미미해지는 문제가 발생할 수 있습니다. 그러나 sine & cosine 함수의 경우 -1 ~ 1 사이를 주기적으로 반복하기에 긴 문장의 Sequence가 주어진다 해도 위치 vector값의 차가 작지 않게됩니다.

3. 같은 위치의 token은 항상 같은 위치 vector값을 가지고 있어야합니다. 그러나 서로 다른 위치의 token은 위치 vector값이 서로 같지않아야합니다.
    여기서의 문제는 -1 ~ 1 사이를 반복하는 주기함수기 때문에 token들의 위치 vetor값이 같은 경우가 생길 수 있습니다.

![캡처](https://user-images.githubusercontent.com/44185037/210730466-10d9b4b4-79b3-402e-bca0-2d76d09da85e.JPG)

* 예를들어 위의 그림과 같이 Sine 함수가 주어진다면 첫 번째 token[P0]과 아홉 번째 token[P8]의 경우 위치 vector값이 같아지는 문제가 발생합니다.

![image (1)](https://user-images.githubusercontent.com/44185037/210731230-c82276c8-2584-4bf2-8bda-b8a715a13449.gif)

* 따라서 위치 vector값이 같아지는 문제를 해결하기 위해 다양한 주기의 sine & cosine 함수를 동시에 사용합니다.
* 하나의 위치 vector가 4개의 차원으로 표현된다면 각 요소는 서로 다른 4개의 주기를 갖게 되기때문에 서로 겹치지않습니다.
* 즉 단어 vector는 각각의 차원마다 서로 다른 위치 vetor값을 가지게 됩니다.

* 위 그림처럼 첫 번째 차원의 vector값들의 차이가 크지않다면, 단어 vector의 다음 차원에도 vector 값을 부여하면됩니다.
* 이때 동일한 sine 값을 사용하게 되면 vector들 간의 차이가 크지 않게 되므로, cosine 값을 사용합니다. 하지만 두 번째 vetor값들 역시 그 차이가 크지 않다면, 서로 다른 단어 vetor간의 위치 정보 차이가 미미하게 됩니다. 이 경우 cosine의 frequency를 이전 sine 함수보다 크게 주면되고
* 마지막 차원의 vector값이 채워질 때 까지 서로 다른 frequency를 가진 sine & cosine을 번갈아가며 계산하다 보면 결과적으로 충분히 서로다른 positional encoding 값을 지니게 됩니다.

![캡처](https://user-images.githubusercontent.com/44185037/210732007-32b5b60a-5585-45f7-ad1b-7cf84695105a.JPG)

* 해당 내용을 수식으로 표현하면 위의 그림과 같습니다.

## Attention 이란?
![image](https://user-images.githubusercontent.com/44185037/210756882-4758c064-17a0-41ac-b56a-2bea398b0d6c.png)

* attention 함수는 유사도를 구하는 함수입니다.
* 주어진 query에 대해서 모든 key와의 유사도를 각각 구하고, 이를 가중치로하여 key와 매핑되어 있는 각각의 값에 반영합니다.
* 이후 유사도가 반영된 값을 모두 가중합하여 최종 리턴합니다(Attention Value)

- 전체 시점에서의 일반화한 Q,K,V의 정의입니다.
    * Q = Querys : 모든 시점의 디코더 셀에서의 은닉 상태들
    * K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
    * V = Values : 모든 시점의 인코더 셀의 은닉 상태들

## self-Attention
* attention을 자기 자신에게 수행한다는 의미입니다.
* 기존의 Q,K,V의 경우에는 Q와 K가 서로 출처를 가지고 있으나 self Attention에서의 Q,K,V는 전부 동일합니다.
- self Attention의 Q,K,V 정의입니다.
    * Q : 입력 문장의 모든 단어 벡터들
    * K : 입력 문장의 모든 단어 벡터들
    * V : 입력 문장의 모든 단어 벡터들

- 여기에서 하나의 예시로 Self-Attention의 기능을 보고가겠습니다.

![image](https://user-images.githubusercontent.com/44185037/210758823-a5b89c17-5331-4d1d-a72d-2254281727f8.png)

* 위의 그림에서 보면 '그 동물은 길을 건너지 않았다. 왜냐하면 너무 피곤했기 때문에다.' 라고 번역할 수 있습니다.
* 사람의 경우 여기서는 'it'이 **동물(animal)**이라고 바로 알 수 있지만
* 여기서 'it'이 **길(street)**인지 **동물(animal)**인지 machine은 어떻게 알 수 있을까요?
* self attention은 입력 문장 내의 단어들끼리 유사도를 구하면서 'it'이 동물과 연관되어 있을 확률이 높다는 것을 찾아냅니다.

- Transformer에서의 self-attention 동작 mechanism
    * self-attention은 위의 예시들 처럼, 단어들 사이의 유사도를 구할 때 주로 이용하기 때문에 입력 문장의 단어 vectors를 가지고 계산합니다.
    * Transformer에서의 self-attentio 동작 mechanism을 이해하기 위해 먼저 input을 생각해보겠습니다.
    * positional encoding을 거치면 위치 정보가 포함된 최종 input embedding matrix가 만들어집니다.
    * d_model 차원의 초기 입력 input embedding matrix 단어 vector들을 바로 사용하는 것이 아니라 각 단어 vectors로 부터 먼저 Q,K,V vectors를 얻어야합니다.

## Q, K, V vector 얻기
* 해당 Q, K, V vectors는 초기 입력의 d_model 차원의 단어 vectors보다 더 작은 차원을 갖습니다.
    -Transformer 논문에서는 d_model=512의 차원을 가졌던 초기 단어 vectors에서 64차원의 Q, K, V vector로 변환하여 사용합니다.
    -d_model/(num_heads) = Q, K, V vector의 차원
    -transformer 논문에서는 num_heads를 8로 하여 512/8 = 64로 Q, K, V vector의 차원을 결정합니다.

![image](https://user-images.githubusercontent.com/44185037/210778190-a2284f4f-3e71-4f5a-bbae-8db4bda141fe.png)

* 기존의 512 차원의 벡터로부터 더 작은 vector Q, K, V 를 만들기 위해서는 가중치 행렬을 곱해야 합니다.
* 가중치 행렬의 크기 : d_model * (d_model / num_heads) 
    -논문의 경우에서는 512 * 64
    -그럼 (1 * 512) x (512 * 64) = (1 * 64) 로 더 작은 벡터인 Q, K, V 구할 수 있음.
    -이 때, Q, K, V vector를 만들기 위한 가중치 행렬은 각각 다릅니다.
    -따라서 512 크기의 하나의 student 단어 벡터에서 서로 다른 3개의 가중치 행렬을 곱해서 64 크기의 서로 다른 3개의 Q, K, V vector를 얻습니다.
    -이 가중치 행렬은 훈련 과정에서 계속 학습됩니다.
* "I am a student." 문장의 모든 단어 벡터에 위의 과정을 거치면, I, am, a, student 단어 각각에 대해서 서로 다른 Q, K, V vectors를 구할 수 있습니다.



## Scaled dot-product Attention 수행
* 현재까지 Q, K, V 벡터들을 얻은 상황. 이제 기존의 어텐션 메커니즘과 동일한 작업 수행
* 먼저 각 Q 벡터는 모든 K 벡터들에 대해서 어텐션 스코어 구하고 - 어텐션 분포를 구하고 - 어텐션 분포로부터 가중치를 적용해 모든 V 벡터들을 가중합을 구하고 - 최종 어텐션 값 ( = context vector) 구합니다. [자세한 내용 https://wikidocs.net/22893 참조]
* 위의 과정을 모든 Q vector에 대해 반복
* Attention Score를 구하기 위한 Attention 함수는 가장 기본인 내적 dot product 이외에도 종류가 다양합니다. [scaled dot, general, concat, location-base]
* Transformer 논문에서는 기본 내적에다가 특정 값으로 나누어주는 Attention 함수(Scaled dot-product Attention)를 사용합니다.

![image](https://user-images.githubusercontent.com/44185037/210779353-d4fd4a29-fd30-41de-9eb0-3dad3521d782.png)
* 위의 수식과 같이 특정값으로 나누어주는 것을 기존의 내적에서 값을 scaling했다고 표현하며, Scaled dot-product Attention이라고 합니다.

- Transformer에는 총 세 가지의 Attetion이 사용됩니다.

![캡처](https://user-images.githubusercontent.com/44185037/210739551-118fa128-8242-474d-88ac-e7821c336b6e.JPG)

1. Encoder Self-Attention
    * encoder에서 이루어집니다.
    * self-attention : query, key, value가 같습니다. [vetor의 값이 같은 것이 아니라 vector의 출처가 같습니다.]
    * encoder의 self-attention : query = key = value

2. Masked Decoder Self-Attention
    * decoder에서 이루어집니다.
    * decoder의 masked self-attention : query = key = value

3. Encoder-Decoder Attention
    * decoder에서 이루어집니다.
    * decoder의 encoder-decoder attention : query = decoder vetor , key=value : encoder vetor

![image](https://user-images.githubusercontent.com/44185037/210740768-e3a1f555-c514-42d2-ac9a-9a836f9dd99d.png)

* 위의 세가지 Attention은 Transformer안에서 위의 그림과 같이 위치해있습니다.
* 세 가지 Attention에 추가적으로 **multi-head**가 붙어있는데, 이는 Transformer attention을 병렬적으로 수행하는 방법을 의미합니다.
