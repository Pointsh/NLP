# Levenshtein-Distance
* 이번에는 NLP에서 문장을 서로 비교하는 편집거리 알고리즘인 Levenshtein-Distance에 대해 알아보겠습니다.

* 우선 아래의 3개의 문자열을 보겠습니다.

1. 나는 너를 좋아해!
2. 너는 나 &nbsp;&nbsp;&nbsp;&nbsp;좋아하니?
3. 오늘 집에  갈거야

* 직관적으로 보면 '나는 너를 좋아해' 와 '너는 나 좋아하니?'가 유사하고 '오늘 집에 갈거야'와는 유사하지않다는 것을 알 수 있습니다.

* 그렇다면 컴퓨터는 이 문장들의 유사도를 어떻게 알 수 있을까요?

* 그래서 **레벤슈타인 거리(Levenshtein Distance)** 에 대해 알아보겠습니다.

1. 레벤슈타인 거리
![image](https://user-images.githubusercontent.com/44185037/213903138-49150656-f32b-4adb-8870-71114be8ce6c.png)

* 위의 식이 레벤슈타인 거리의 정리된 식인데 해당 식만 보고는 어려우니 예시로 표현해보겠습니다.

![image](https://user-images.githubusercontent.com/44185037/213903161-dca5aed1-76bf-4317-9af2-69162e18a5b9.png)

* 두 문자열을 비교하면 문자 조작 비용은 총 6입니다.
* 수정1 수정1 삭제1 수정1 수정1 삽입1 = 6 

* 위의 조작 비용이 어떻게 결정되는지 알고리즘을 통해 알아보겠습니다.
![image](https://user-images.githubusercontent.com/44185037/213903347-76573cc5-7cd6-4883-a684-91fd20b31679.png)

* 처음의 비교 대상은 공집합과 공집합입니다. 둘다 같은 문자열이기에 비용은 0입니다.
* 그 다음은 공집합과 '나' 입니다. 공집합이 '나'가 되려면 비용은 수정1입니다.
* 그 다음은 공집합과 '나는'입니다. 공집합이 '나는'이 되려면 수정1삽입1 이므로 총 비용 2가 듭니다.
* 이와 같이 계속 진행한다면 위의 표와 같이 숫자들이 비용들이 배치되게 됩니다.
# 전체적인 표 비용 계산
* 이제 '너'와 '나' , '나는', '나는 '...'나는 너를 좋아해!'를 비교해보겠습니다.
* '너'와 공집합을 보면 '너'가 공집합이 되려면 문자를 삭제해야합니다. 삭제 비용1
* '너'와 '나'를 보면 서로 다르기에 수정비용 1
* '너'와 '나는'을 보면 길이가 다르기에 삽입 비용1 서로 다르기에 수정 비용1 총 비용 2
* '너'와 '나는 너'를 보면 길이가 다르기에 띄어쓰기 포함 3개를 추가하고 '너'는 같기에 그대로둬서 총 비용 삽입 비용3
* 이와 같이 '나는 너를 좋아해!'까지 표를 완성하면 아래와 같이 됩니다.
![image](https://user-images.githubusercontent.com/44185037/213903754-f9788530-4bad-4be8-ab43-f2856066004f.png)

* 이런식으로 표를 완성해나가면 아래와 같은 표가 완성됩니다.
![image](https://user-images.githubusercontent.com/44185037/213903801-619e1535-042d-430c-8cd9-e732e989fd2b.png)

# 간단한 공식
* 간단한 공식으로 표현하자면
1. 글자가 서로 동일하면 대각선 값을 가져옵니다.
2. 변경이 필요하면 대각선 값에서 + 1을 합니다.
3. 삽입이 필요하면 위의 값에서 +1을 합니다.
4. 삭제가 필요하면 왼쪽 값에서 +1을 합니다.
5. 1~4의 경우에서 최소값을 가져옵니다.

* 이를 python code로 표현하면 아래와 같습니다.
```python
import numpy as np
def leven(text1,text2):
    len1 = len(text1)+1
    len2 = len(text2)+1
    sim_array = np.zeros((len1,len2))
    sim_array[:,0] = np.linspace(0, len1-1, len1)
    sim_array[0,:] = np.linspace(0, len2-1, len2)
    for i in range(1,len1):
        for j in range(1,len2):
            add_char = sim_array[i-1,j]+1
            sub_char = sim_array[i,j-1]+1
            if text1[i-1] == text2[j-1]:
                mod_char = sim_array[i-1,j-1]
            else:
                mod_char = sim_array[i-1,j-1]+1
            sim_array[i,j] = min([add_char, sub_char, mod_char])
    return sim_array[-1,-1]
print(leven('나는 너를 좋아해','너는 나   좋아하니'))
```
![image](https://user-images.githubusercontent.com/44185037/213904149-50986c97-ab7d-4b8f-a343-a70c8f09b2fe.png)
* 이와같이 결과가 나오게 됩니다.
