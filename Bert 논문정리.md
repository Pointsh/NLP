BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding
===============================================================================

## 해당 페이지의 구성은 논문을 쭉 읽어나가며 공부하고 정리한 포스트입니다.


- BERT : Bidirectional Encoder Representations form Transformer
해당 논문의 제목입니다. 해당 논문은 “Attention is all you need(Vaswani et al., 2017)”에서 소개한 **Transformer** 구조를 활용한 언어표현에 관한 논문이라고합니다.
Transformer에 대한 구조는 아래와 같습니다.

<div align='center'>![트랜스포머 구조](https://user-images.githubusercontent.com/44185037/210715269-a08e3167-1f4d-42b1-93e8-1a3eca136a0e.JPG)</div>

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




