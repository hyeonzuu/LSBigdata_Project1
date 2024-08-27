import numpy as np

# 벡터 * 벡터 (내적)
a = np.array([3, 6, 9])
a.shape
b = np.arange(1, 4)
b

a.dot(b) #dot 벡터 내적 계산

# 행렬 * 벡터
a = np.array([1, 2, 3, 4]).reshape((2, 2), order="F") # order="F" 세로로 쌓음
a
b = np.array([5, 6])
b

a.dot(b)
a @ b # 행렬 계산 위에 거랑 똑같음

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2), order="F")
b = np.array([5,6,7,8]).reshape((2, 2), order="F")
a @ b 

#Q1
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b = np.array([1, 0,-1,1,2,3]).reshape(3, 2)
a @ b 

#Q2
i = np.eye(3) # identity metrix
a = np.array([3,5,7,
          2,4,9,
          3,1,0]).reshape(3, 3)
i @ a
a @ i

# transpose
a
a.transpose()
b=a[:,0:2]
b
b.transpose()

#-----------------
x = np.array([13, 15,
          12, 14,
          10, 11,
          5, 6]).reshape(4, 2)
vec1 = np.repeat(1, 4).reshape(4, 1)
matx = np.hstack((vec1, x))
matx

beta_vec = np.array([2,0,1]).reshape(3, 1)

y_hat = matx @ beta_vec
y_hat

y = np.array([20, 19, 20, 12]).reshape(4, 1)

(y - y_hat).transpose() @ (y - y_hat)


# 역행렬
a = np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv = 1/(4-15) * np.array([4,-5,-3,1]).reshape(2,2) # 2*2 역행렬 공식
a@a_inv

## 3by3 역행렬
a = np.array([-4, -6, 2,
              5, -1, 3,
              -2, 4,-3]).reshape(3, 3)

a_inv = np.linalg.inv(a)
a_inv

np.round(a @ a_inv, 3)

b = np.array([1, 2, 3,
              2, 4, 5,
              3, 6, 7]).reshape(3, 3)

b_inv = np.linalg.inv(b)
np.linalg.det(b) # 선형종속인지 아닌지. =0나오면 선형 종속

# beta_hat
XtX_inv = np.linalg.inv((matx.transpose() @ matx))
Xty = matx.transpose() @ y
beta_hat = XtX_inv @ Xty

from sklearn.linear_model import LinearRegression 
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(matx[:, 1:], y)

model.coef_   
model.intercept_ 


# minimize로 베타 구하기
from scipy.optimize import minimize
def line_perform(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matx @ beta)
    return (a.transpose() @ a)

# line_perform([6, 1, 3])
# line_perform([ 8.55,  5.96, -4.38])

# 초기 추정값
initial_guess = [0, 1, 0]
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)
# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matx @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum() #abs 절댓값

# line_perform([8.55,  5.96, -4.38])
# line_perform([3.76,  1.36, 0])
# line_perform_lasso([8.55,  5.96, -4.38])
# line_perform_lasso([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matx @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55,  5.96, -4.38])
line_perform_ridge([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize
def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum()
# 초기 추정값
initial_guess = [0, 0, 0]
# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)
# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

[8.14, 0.96, 0]
# 예측식: y_hat = 8.14 + 0.96*X1 + 0 * X2
[8.55,  5.96, -4.38] # 람다 0

[8.14,  0.96, 0] # 람다 3
# 예측식: y_hat = 8.14 + 0.96 * X1 + 0 * X2

[17.74, 0, 0] # 람다 500
# 예측식: y_hat = 17.74 + 0 * X1 + 0 * X2

# 람다값에 따라 변수 선택된다.
# x 변수가 추가되면 trainX에서는 성능이 항상 좋음.
# x 변수가 추가되면 valid에서 좋아졌다가 나빠짐.(오버피팅)
# 어느 순간 x 변수 추가를 중단함.
# 람다 0부터 시작 : 내가 가진 모든 변수를 넣겠다.
# 점점 람다가 증가하면 변수가 하나씩 빠지게됨.
# 따라서 람다가 증가하다가 valid에서 성능이 가장 좋은 람다 선택
# 변수가 선택됨을 의미

# X의 칼럼에 선형종속인 애들이 있다 : 다중공선성이 존재한다.(beta 추정이 되어도 믿을 수가 없다)

