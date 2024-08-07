import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t

# y = 2x + 3
x = np.linspace(0, 100, 400)
y = 2*x + 3

np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i = norm.rvs(loc=0, scale=20, size=20)

# yi 식 코드
obs_y = 2*obs_x +3 + epsilon_i
--------------------------------------------------
# 예제, 아래 코드에서 a,b 추정 후 그래프 그리기
df = pd.DataFrame({"x": obs_x,
                   "y": obs_y})
                   
model = LinearRegression()

obs_x = obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)

a = model.coef_[0] # 기울기 a 
b = model.intercept_ # y절편 b


# 그래프 그리기
plt.plot(x, y, color='black') # y = 2x + 3 그래프
plt.scatter(obs_x, obs_y, color="blue", s= 3) 
plt.plot(df["x"], a * df["x"] + b, color = "red") # a,b 직선 , 데이터로부터 추정한 직선
plt.show()
plt.clf()

# n이 커지고 epsilon_i scale이 커질 수록 빨간선과 검정선이 겹쳐짐

#!pip install statsmodels

import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())


1- norm.cdf(18, loc=10, scale=1.96)

1- norm.cdf(4.08, loc=0, scale=1)

# 교재 57p
# 2. 검정을 위한 가설을 명확하게 서술하시오. 
x = np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,
15.382, 16.709, 16.804])
n = 15

x_bar = np.mean(x)
x_std = np.std(x)
# 3. 검정통계량을 계산하시오
t = (x_bar - 16) / (x_std / np.sqrt(15))

# 4. p‑value을 구하세요.
p_value = norm.cdf(x_bar, loc=16, scale=(x_std / np.sqrt(15)))

# 6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.
se = np.std(x, ddof=1) / np.sqrt(n)
round(x_bar - t.ppf(0.975, n-1) * se, 3)
round(x_bar + t.ppf(0.975, n-1) * se, 3)
