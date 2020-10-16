from collections import deque

de=deque(maxlen=3)
for i in range(10):
    de.append(i)
print(de)