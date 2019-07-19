import numpy as np

turn = np.zeros(14, dtype=np.float32)
turnTmp = "00000000000000"
ans = 4
if ans < 14:
    turnTmp = turnTmp[:ans] + "1"+turnTmp[ans+1:]
print(turnTmp)

for i in range(14):
    turn[i] = float(turnTmp[i])
print(turn)
