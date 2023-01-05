BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding
===============================================================================

## 해당 페이지의 구성은 논문을 쭉 읽어나가며 공부하고 정리한 포스트입니다.


- BERT : Bidirectional Encoder Representations form Transformer
해당 논문의 제목입니다. 해당 논문은 “Attention is all you need(Vaswani et al., 2017)”에서 소개한 **Transformer** 구조를 활용한 언어표현에 관한 논문이라고합니다.
Transformer에 대한 구조는 아래와 같습니다.

![트랜스포머 구조](https://user-images.githubusercontent.com/44185037/210715269-a08e3167-1f4d-42b1-93e8-1a3eca136a0e.JPG)

- 트랜스포머는 Attention 기법만 사용하기때문에 RNN,CNN은 전혀 사용하지않습니다.
그러나 기존의 **seq2seq**의 인코더-디코더 구조는 유지합니다.
인코더에서 입력 시퀀스를 입력받고, 디코더에서 출력 시퀀스를 출력합니다.

구조가 위와 같기때문에 문장 안에 포함된 각각의 단어들의 순서에 대한 정보를 주기 어렵습니다.
문장내 각각 단어의 순서에 대한 정보를 알려주기 위해 **Positional encoding**을 사용합니다
이러한 아키텍처는 **BERT**와 같은 향상된 네트워크에서도 채택되었습니다.
Attention 과정 한번만 사용하는 것이 아니라 여러 Layer를 거쳐서 반복하도록 구현되어있습니다.

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
위의 그림은 하나의 단어를 vetor화 시킨 모습입니다. 그 옆에 그림과 같이 Positional Encoding을 통해 얻은 위치정보를 더해주야합니다.
이때 지켜야 될 규칙 두가지가 있습니다.

1. 모든 위치값은 sequence의 길이나 input에 관계없이 동일한 식별자를 가져야합니다. 따라서 sequence가 변경되더라도 위치 embedding은 동일하게 유지됩니다.
![캡처](https://user-images.githubusercontent.com/44185037/210728014-67a5159d-0b27-493d-9059-183dfdeef26c.JPG)

2. 모든 위치값은 너무 크면 안됩니다. 위치값이 너무 커져버리면 단어간의 상관관계 및 의미를 유추할 수 있는 의미정보 값이 상대적으로 작아지게되고, Attention layer에서 제대로 학습 및 훈련이 되지 않을 수 있습니다.

![캡처](https://user-images.githubusercontent.com/44185037/210728485-ec6d4641-a82b-4a51-a797-352860849126.JPG)


- 위의 두 가지 규칙을 지키면서 vetor를 부과하는 방법에는 sine & cosine 함수가 있습니다.

해당 sine & cosine 함수가 positional encoding에서 사용한 이유는 총 3가지가 있습니다.

1. 의미정보가 변질되지 않도록 위치 vetor값이 너무 크면 안된다.
- sine & cosine 함수는 -1 ~ 1 사이를 반복하는 주기함수입니다. 즉 1을 초과하지않고 -1 미만으로 떨어지지 않으므로 값이 너무 커지지 않는 조건을 만족시킵니다.

2. sine & cosine 함수 외에도 일정 구간 내에 있는 함수로는 Sigmoid 함수가 있습니다. 해당 함수를 채택하지않고 sine & cosine 함수를 선택한 이유는 **주기함수**이기 떄문입니다.
![캡처](https://user-images.githubusercontent.com/44185037/210729740-45c6d201-407f-41e0-bb75-bde349a6bc15.JPG)
- 위의 그림과 같이 Sigmoid 함수의 경우, 긴 문장의 sequence가 주어진 경우, 위치 vetor 값의 차가 미미해지는 문제가 발생할 수 있습니다. 그러나 sine & cosine 함수의 경우 -1 ~ 1 사이를 주기적으로 반복하기에 긴 문장의 Sequence가 주어진다 해도 위치 vetor값의 차가 작지 않게됩니다.

3. 같은 위치의 token은 항상 같은 위치 vetor값을 가지고 있어야합니다. 그러나 서로 다른 위치의 token은 위치 vetor값이 서로 같지않아야합니다.
여기서의 문제는 -1 ~ 1 사이를 반복하는 주기함수기 때문에 token들의 위치 vetor값이 같은 경우가 생길 수 있습니다.

![캡처](https://user-images.githubusercontent.com/44185037/210730466-10d9b4b4-79b3-402e-bca0-2d76d09da85e.JPG)
예를들어 위의 그림과 같이 Sine 함수가 주어진다면 첫 번째 token[P0]과 아홉 번째 token[P8]의 경우 위치 vetor값이 같아지는 문제가 발생합니다.

![image (1)](https://user-images.githubusercontent.com/44185037/210731230-c82276c8-2584-4bf2-8bda-b8a715a13449.gif)
따라서 위치 vetor값이 같아지는 문제를 해결하기 위해 다양한 주기의 sine & cosine 함수를 동시에 사용합니다.
하나의 위치 vetor가 4개의 차원으로 표현된다면 각 요소는 서로 다른 4개의 주기를 갖게 되기때문에 서로 겹치지않습니다.
즉 단어 vetor는 각각의 차원마다 서로 다른 위치 vetor값을 가지게 됩니다.

위 그림처럼 첫 번째 차원의 vetor값들의 차이가 크지않다면, 단어 veotr의 다음 차원에도 vetor 값을 부여하면됩니다.
이때 동일한 sine 값을 사용하게 되면 vetor들 간의 차이가 크지 않게 되므로, cosine 값을 사용합니다.
하지만 두 번째 vetor값들 역시 그 차이가 크지 않다면, 서로 다른 단어 vetor간의 위치 정보 차이가 미미하게 됩니다. 이 경우 cosine의 frequency를 이전 sine 함수보다 크게 주면되고
마지막 차원의 vetor값이 채워질 때 까지 서로 다른 frequency를 가진 sine & cosine을 번갈아가며 계산하다 보면 결과적으로 충분히 서로다른 positional encoding 값을 지니게 됩니다.

![캡처](https://user-images.githubusercontent.com/44185037/210732007-32b5b60a-5585-45f7-ad1b-7cf84695105a.JPG)
해당 내용을 수식으로 표현하면 위의 그림과 같습니다.