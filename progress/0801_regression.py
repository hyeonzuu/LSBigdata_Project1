import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# y = 2x + 3의 그래프를 그려보세요
a = 2
b = 3
x = np.linspace(-5, 5, 100)
y = a*x +b

plt.plot(x, y)
plt.axvline(0, color = "black",linewidth=1)
plt.axhline(0, color = "black",linewidth=1)
plt.xlim(-5, 5)
plt.ylim(-5,5)
plt.show()
plt.clf()

# 
# # homeprice
# a= 80
# b = 5
# x = np.linspace(0, 5, 100)
# y = a*x +b
# 
# house_train = pd.read_csv("data/house price/train.csv")
# my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
# my_df["SalePrice"] = my_df["SalePrice"] / 1000
# plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
# plt.plot(x,y, color="black")
# plt.show()
# plt.clf()



a= 63
b = 100
x = np.linspace(0, 5, 100)
y = a*x +b

house_train = pd.read_csv("data/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]]
my_df["SalePrice"] = my_df["SalePrice"] / 1000
my_df["SalePrice"] = my_df["BedroomAbvGr"]*(my_df["SalePrice"].mean())+100
plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
plt.plot(x,y, color="black")
plt.show()
plt.clf()


test = pd.read_csv("data/test.csv")
test = pd.merge(test, my_df, how="left", on ="BedroomAbvGr")
test["SalePrice"] = test["SalePrice"] * 1000

sub = pd.read_csv("data/sample_submission.csv")
sub["SalePrice"] = test["SalePrice"]
sub.to_csv("sub_prediction08011.csv", index=False)
-----------------------------------------------------------------------------
# 테스트 집 정보 가져오기
sub = pd.read_csv("data/sample_submission.csv")
a = 63
b = 100
(a * test["BedroomAbvGr"] + b) * 1000

sub["SalePrice"] = my_df["SalePrice"]*1000

sub.to_csv("sub_prediction08.csv", index=False)

----------------------------------------------------------------------------
# 직선 성능 평가
a = 70
b = 10

# y_hat 어떻게 구할까
train = pd.read_csv("data/train.csv")
y_hat = (a * train["BedroomAbvGr"] + b) *1000
# y는 어디에 있는가
y = train["SalePrice"]

np.abs(y - y_hat) # abs :절대값 
np.sum(np.abs(y - y_hat)) #  직선 y = 70x + 10의 성능값 nnp.int64(106021410)
-----------------------------------------------------------------------------
!pip install scikit-learn


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편값을 구함 ?어떻게 구하지?

# 회귀 직선의 기울기와 절편
 model.coef_ # 기울기 a 
 model.intercept_# 절편 b
 
slope = model.coef_[0] # coef : 계수
intercept = model.intercept_  
print(f"slope (slope): {slope}")
print(f"intercept (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x) # x 는 train의 방정보

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

-------------------------------------------------------------------------------
# 회귀 모델을 활용한 집값 예측
train = pd.read_csv("data/train.csv")
train = train[["BedroomAbvGr", "SalePrice"]]
x = train["BedroomAbvGr"].to_numpy().reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = train["SalePrice"]  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편값을 구함 ?어떻게 구하지?

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 a 
model.intercept_# 절편 b

slope = model.coef_[0] # coef :계수
intercept = model.intercept_  
print(f"slope (slope): {slope}")
print(f"intercept (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()




