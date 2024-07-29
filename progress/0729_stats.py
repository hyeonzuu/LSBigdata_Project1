import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# y=2x
# 점을 직선으로 이어서 표현현
y = 2 * x
x = np.linspace(0, 8, 2)

plt.scatter(x, y, s =2)
plt.plot(x, y)
plt.show()
plt.clf()

# y = x^2 점세개를 사용해서서

x = np.linspace(-8, 8 100)
y = x ** 2

# plt.scatter(x, y,s =3 )
plt.plot(x, y, color="black")
plt.xlim(-10, 10)
plt.ylim(0, 40)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
plt.clf()

--------------------------------------------------------------
from scipy.stats import uniform
from scipy.stats import binom
from scipy.stats import norm
# 1. X bar, n, sigma, Z
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean() # X bar = 68.893
len(x) # n =6 
# sigma = 6 
# alpha = 0.1
# Zalpha/2 = Z 0.05 정규분포

Z_005 = norm.ppf(0.95, loc=0, scale=1)
Z_005
x.mean() + Z_005 * 6 / np.sqrt(16)
x.mean() - Z_005 * 6 / np.sqrt(16)

______________________________________________________________

# 표본분산
# 데이터로 부터 E[X^2] 구하기
x = norm.rvs(loc=3, scale=5, size = 100000)
np.mean(x**2)
sum(x**2) / (len(x) - 1)

# E[(X - X^2)/2X]
np.mean((x - x**2) / (2*x))

# => 몬테카를로 적분 

# 
np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size = 100000)
x_bar = x.mean()
(x - x_bar) **2
s_2 = sum((x - x_bar)**2) / (100000- 1)

np.var(x, ddof = 1) # n-1로 나눈값(표본분산)
np.var(x, ddof = 0) # n으로 나눈값, 사용 X

# n-1 vs n
x = norm.rvs(loc=3, scale=5, size = 100000)
np.var(x)
np.var(x, ddof=1)



















