s = 'banana'
from collections import deque
q = deque(s)

while q:
    n = q.popleft()
    print(n)