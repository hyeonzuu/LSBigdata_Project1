import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize

train = pd.read_csv("data/train.csv")
train = train[["Id", "OverallCond", "OverallQual","SalePrice"]]
x = (train["OverallCond"] + train["OverallQual"]).to_numpy().reshape(-1, 1)
 # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
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
pred_y = model.predict(x)

test = pd.read_csv("data/test.csv")

test_x =  (test["OverallCond"] + test["OverallQual"]).to_numpy().reshape(-1, 1) #  test 셋에 대한 집값
pred_y = model.predict(test_x)

sub = pd.read_csv("data/sample_submission.csv")

# SalePrice 바꿔치기
sub["SalePrice"] = pred_y
sub.to_csv("sub_prediction0802_2.csv", index=False)

