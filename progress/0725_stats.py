from scipy.stats import binom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

binom.pmf(1, 0.3)
binom.pmf(0, 0.3)


# 이항 분포 X ~ P(X = k \ n, p)
# n : 베르누이 확률변수 
# p : 1이 나올 확률
# binom.pmf(k, n, p)
binom.pmf(0, n=2,p = 0.3)
binom.pmf(1, n=2,p = 0.3)
binom.pmf(2, n=2,p = 0.3)

# X ~ B(n, p)
binom.pmf(np.arange(31), n = 30, p = 0.3)

math.factorial(54) / math.factorial(26) * math.factorial(28)
math.comb(54, 26)

==============================================

# 위 함수 없이 계산
# 아래는 내가 한거
a = np.cumprod(np.arange(54, 27, -1))[-1]
b = np.cumprod(np.arange(1, 26))[-1]
a / b
# ln
log(a * b) = log(a) +lob(b)
log(1 * 2 * 3 * 4) = log(1) +log(2) + log(3) + log(4)

math.log(24)
sum(np.log(np.arange(1 ,5)))

math.log(math.factorial(54))
log_54 = sum(np.log(np.arange(1 ,55)))
log_28 = sum(np.log(np.arange(1 ,29)))
log_26 = sum(np.log(np.arange(1 ,27)))

log_54 - (log_28 + log_26)
np.exp(35.168)
============================================

math.comb(2, 0) * 0.3**0  * (1-0.3) ** 2
math.comb(2, 1) * 0.3**1  * (1-0.3) ** 1
math.comb(2, 2) * 0.3**2  * (1-0.3) ** 0

# pmf : probability mass fucntion(확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

#Q1
binom.pmf(x, n=10, p=0.36)
np.arange(5)
binom.pmf(np.arange(5), n=10, p=0.36).sum()
np.arange(3, 9)
binom.pmf(np.arange(3, 9), n=10, p=0.36).sum()

# 내가한거 틀림
# x = np.arange(0, 31)
# binom.pmf(x< 4 | x >=25, n=30, p=0.2).sum()

#X ~ B(30, 0.2)
# x < 4 | x >= 25
#1
1 - binom.pmf(np.arange(4, 24), n=30, p=0.2).sum()
#2
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
b = binom.pmf(np.arange(25, 31), n=30, p=0.2).sum()
a + b

# rvs(Random Variates Sample) 표본 추출 함수
bernoulli.rvs(p=0.3)
bernoulli.rvs(p=0.3) + bernoulli.rvs(p=0.3)
binom.rvs(n=2, p=0.3, size=1)

# X ~ B(30, 0.26)
# 표본 30개를 뽑아보세요
binom.rvs(n=30, p=0.26, size=30)

# X ~ B(30, 0.26)
binom.rvs(n=30, p=0.26, size=30)
30 * 0.26

# X ~ B(30, 0.26) 시각화

x = np.arange(31)
prob_x = binom.pmf(np.arange(31), n=30, p=0.26)

sns.barplot(prob_x)
plt.show()
plt.clf()

df = pd.DataFrame({"x": x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()
plt.clf()

# cdf(cumulative dist. fucntion) 누적확률밀도
# F(X=x) = P(X <= x)
binom.cdf(x,n,p)
binom.cdf(4, n=30, p=0.26)

# P(4 < x <= 18)
a = binom.cdf(4, n=30, p=0.26)
b = binom.cdf(18, n=30, p=0.26)
b - a #np.float64(0.9213095953839117)

# P(13<x<20)
a = binom.cdf(13, n=30, p=0.26)
b = binom.cdf(19, n=30, p=0.26)
b - a

x_1 = binom.rvs(n=30, p=0.26, size=10)
x = np.arange(31)
prob_x = binom.pmf(np.arange(31), n=30, p=0.26)
sns.barplot(prob_x, color="pink")

plt.scatter(x_1, np.repeat(0.002,10), color='red', zorder=100, s=5)

# 기대값 표현
plt.axvline(7.8, 0.002, color='blue', linestyle = "--", linewidth=2)


plt.show()
plt.clf()

# PPF
binom.ppf(0.5, n = 30, p=0.26)
binom.cdf(8, n = 30, p=0.26)
binom.cdf(7, n = 30, p=0.26)
# P(X < ?) = 0.7
binom.ppf(0.7, n = 30, p=0.26)
binom.cdf(9, n = 30, p=0.26)

1/np.sqrt(2 * math.pi)
from scipy.stats import norm

# mu:loc, sigma:scale
norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)

# 정규분포는 원하는 범위 설정해야함
k =np.linspace(-3, 3, 100)
y = norm.pdf(k, loc=0, scale=1)
# 정규분포 pdf 그리기
plt.scatter(k, y, color='red', s =1)
plt.show()
plt.clf()

plt.plot(k, y, color='black')
plt.show()
plt.clf()

# mu (loc) : 종분포의 중심 결정하는 모수
k =np.linspace(-3, 3, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color='black')
plt.show()
plt.clf()

# sigma (scale) : 분포의 퍼짐을 결정하는 모수
k =np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)
y2 = norm.pdf(k, loc=0, scale=2)
plt.plot(k, y, color='black')


plt.show()
plt.clf()

















