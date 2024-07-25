import numpy as np
x = np.arange(33)

x.sum()/33

np.unique((x - 16)**2)

sum.(np.unique((x - 16)**2) * (2/33))


# 제곱의 기댓값
sum(x**2 * (1/33))

# Var(x) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2

# 퀴즈
# x = 0,1,2,3
# 1/6, 2/6, 2/6, 1/6
x = np.arange(4)
pro_x = np.array([1/6, 2/6, 2/6, 1/6])

# 기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2

sum((x - Ex)**2 * pro_x)

np.unique((x-1.5)**2)

# 퀴즈2 
# x = 0~98까지 정수

x = np.arange(99) # 확률 변수 x
pro_x = np.concatenate((np.arange(1,51), np.arange(49,0,-1))) / 2500 # x가 나올 확률

Ex = sum(x * pro_x) 
Exx = sum(x**2 * pro_x)

Exx - Ex**2

# 퀴즈3
y = np.arange(0, 7, 2)
y = np.arange(4) * 2

pro_y = np.array([1/6, 2/6, 2/6, 1/6])

Ey = sum(y * pro_y) 
Eyy = sum(y**2 * pro_y)

Eyy - Ey**2

import numpy as np

np.sqrt(9.53**2 / 10)




