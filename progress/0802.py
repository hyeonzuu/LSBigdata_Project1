import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# y = x^2 + 3 
def my_f(x):
    return x**2 + 3
my_f1(3)

# 초기 추정값
initian_guess = [0]

result = minimize(my_f,initian_guess )
print("최소값",result.fun)
print("최소값을 갖는 x 값",result.x)

# y = 
def my_f2(x):
    return x[0]**2 + x[1]**2 + 3

my_f2([1, 3])

initian_guess = [-10,3]

result = minimize(my_f2,initian_guess )
print("최소값",result.fun)
print("최소값을 갖는 x 값",result.x)

# f(x, y, z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7


initian_guess = [-10, 3, 2]

result = minimize(my_f3,initian_guess )
print("최소값",result.fun)
print("최소값을 갖는 x 값",result.x)
---------------------------------------------------------------------------------


train = pd.read_csv("data/train.csv")
train = train[["BedroomAbvGr", "SalePrice"]]
x = train["BedroomAbvGr"].to_numpy().reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = train["SalePrice"] / 1000 # y 벡터 (레이블 벡터는 1차원 배열입니다)

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
test = pd.read_csv("data/test.csv")

test_x = np.array(test["BedroomAbvGr"]).reshape(-1,1) #  test 셋에 대한 집값
pred_y = model.predict(test_x)

sub = pd.read_csv("data/sample_submission.csv")

# SalePrice 바꿔치기
sub["SalePrice"] = pred_y * 1000
sub.to_csv("sub_prediction0802.csv", index=False)

--------------------------------------------------------------------------------
train = pd.read_csv("data/train.csv")
train = train[["GrLivArea", "SalePrice"]]

# 이상치 탐지
train = train.query("GrLivArea <= 4500")
#train['GrLivArea'].sort_values(ascending = False).head(2)

x = train["GrLivArea"].to_numpy().reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = train["SalePrice"] / 1000 # y 벡터 (레이블 벡터는 1차원 배열입니다)


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
test = pd.read_csv("data/test.csv")

test_x = np.array(test["GrLivArea"]).reshape(-1,1) #  test 셋에 대한 집값
pred_y = model.predict(test_x)

sub = pd.read_csv("data/sample_submission.csv")

# SalePrice 바꿔치기
sub["SalePrice"] = pred_y * 1000
sub.to_csv("sub_prediction0802_3.csv", index=False)

-----------------------------------------------------------------------------
# 원하는 변수 2개 사용 다항회귀
train = pd.read_csv("data/train.csv")

# 이상치 탐지
#train = train.query("GrLivArea <= 4500")
#train['GrLivArea'].sort_values(ascending = False).head(2)

# x = train[["GrLivArea", "GarageArea"]].to_numpy().reshape(-1, 2)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
x = train[["GrLivArea", "GarageArea"]]
y = train["SalePrice"] # y 벡터 (레이블 벡터는 1차원 배열입니다)

# y = train["SalePrice"] # 시리즈
# y = train[["SalePrice"]] # 데이터 프레임

y = train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편값을 구함 ?어떻게 구하지?

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a 
model.intercept_ # 절편 b

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

test = pd.read_csv("data/test.csv")

test_x = np.array(test["GrLivArea"]).reshape(-1,1) #  test 셋에 대한 집값
pred_y = model.predict(test_x)

sub = pd.read_csv("data/sample_submission.csv")

# SalePrice 바꿔치기
sub["SalePrice"] = pred_y * 1000
#sub.to_csv("sub_prediction0802_3.csv", index=False)
------------------------------------------------------------------------------------

def f(x,y):
    return model.coef_[0] * x + model.coef_[1] * y + model.intercept_  
f(300, 55)



# 선형 회귀 모델 생성
model = LinearRegression()

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



















