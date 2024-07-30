from scipy.stats import uniform
from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

# uniform에서 loc = 구간 시작점, scale = 구간길이
uniform.rvs(loc=2, scale=4, size = 1)
uniform.pdf(3, loc=2, scale=4)
uniform.pdf(7, loc=2, scale=4)

# uniform pdf 그리기
k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y)
plt.show()
plt.clf()

# P(X < 3.25) 확률?
# 넓이로 확률 구하기 : 1.25 * 0.25
uniform.cdf(3.25, loc=2, scale=4)

# P(5 < X < 8.39) 확률?
uniform.cdf(3, loc=2, scale=4)
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)

# 상위 7%값? 
uniform.ppf(0.93, loc=2, scale=4)

# 표본 20개 뽑아서 표본 평균 계산하기
# random_state=(None | 실수) 랜덤값 지정
uniform.rvs(loc=2, scale=4, size=20, random_state=42).mean()

x = uniform.rvs(loc=2, scale=4, size=20 *1000, random_state=42)

x = x.reshape(1000, 20)# ???
x.shape
blue_x = x.mean(axis=1)

# 히스토그램 시각화
sns.histplot(blue_x, stat = 'density') 
plt.show()

# X bar ~ N(4,sigma^2/n) 
# X bar ~ N(4,1.3333/20) 
# n은 한번 뽑을 때 검은 블록
uniform.var(loc = 2, scale=4) # np.float64(1.3333333333333333)
uniform.expect(loc = 2, scale=4) # np.float64(4.0)

xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color = 'red', linewidth=2)
plt.show()
plt.clf()
# 히스토그램을 그릴 필요가 있는가..? 


# 신뢰 구간 
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color = 'red', linewidth=2)

# 기대값 표현
plt.axvline(x=4, color='green', linestyle = "-", linewidth=2)
# 표본평균 (파란벽돌) 점 찍기
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
plt.scatter(blue_x,0.002, color='blue', zorder=100, s=5)

plt.show()
plt.clf()
# => 정규분포

# 95%에 해당하는 a,b값
4 - norm.ppf(0.025, loc = 4, scale=np.sqrt(1.3333/20)) # 중심에서 0.505 떨어지면 됨
4 - norm.ppf(0.975, loc = 4, scale=np.sqrt(1.3333/20)) # 중심값 = 4
# 99%에 해당하는 a,b값
4 - norm.ppf(0.005, loc = 4, scale=np.sqrt(1.3333/20))# 중심에서 0.665 떨어지면 됨
4 - norm.ppf(0.995, loc = 4, scale=np.sqrt(1.3333/20))

norm.ppf(0.025, loc = 4, scale=np.sqrt(1.3333/20))
norm.ppf(0.975, loc = 4, scale=np.sqrt(1.3333/20))

norm.ppf(0.975, loc=0, scale=1) # 1.96 ????


x = uniform.rvs(loc=2, scale=4, size=20 *1000, random_state=42)
x = x.reshape(1000, 20)
x.shape

# 히스토그램 시각화
sns.histplot(blue_x, stat = 'density') 
plt.show()

uniform.var(loc = 2, scale=4) # np.float64(1.3333333333333333)
uniform.expect(loc = 2, scale=4) # np.float64(4.0)


