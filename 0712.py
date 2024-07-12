a = [1, 2, 3]

# Deep copy
b = a[:]
b = a.copy()
a[1] = 4
a
b

# 숫자와 친해지기
# 수학 함수
import math
X = 4
math.sqrt(X)

# 지수 계산
exp_val = math.exp(5)
exp_val

# 로그 계산
log_x = math.log(10,10)
log_x

# 팩토리얼 계산
fact_x = math.factorial(5)
fact_x

# 삼각함수 
cos_x = math.cos(math.radians(180))
cos_x

# 예제
import math

def normal_pdf(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def my_nomal_pdf(x, mu, sig):
  part1 = 1/(sig * math.sqrt(2 * math.pi))**-1
  part2 = math.exp(-(x - mu)**2 / (2 * (sig)**2))
  return part1 * part2

my_nomal_pdf(3, 3, 3)

#파라미터 
x = 3
mu = 3
sig = 3

def exam(x, y, z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
exam(2, 9, math.pi/2)

def test(x):
  return math.cos(x) + math.sin(x) * math.exp(x)
test(math.pi)

#'fcn' + shift +tab 누르면 나온다
def (input):
    Contents
    return
#'pd' or 'np' + shift +tab 누르면 나온다
import pandas as pd
import numpy as np

# 벡터와 친해지기
import numpy as np

# Ctrl + Shift + C => #

# 벡터 생성
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
a
type(a)
a[3]
a[2:] # 인덱싱해도 어레이 유지
a[1:4]

b = np.empty(3)
b
b[0] = 1
b[1] = 2
b[2] = 3
b[2]

vec1 = np.array([1, 2, 3 ,4, 5])
vec2 = np.arange(3,10, 0.5) #함수를 사용하여 일정한 간격의 숫자 배열 생성
vec2

vec3 = np.linspace(0, 1, 5) #함수를 사용하여 지정된 범위를 균일하게 나눈 숫자 배열 생
vec3

# -100부터 0까지
y = np.arange(0, -100, -1)
z = -np.arange(0, 100)
y
z

# repeat vs tile
vec1 = np.arange(5)
np.repeat(vec1, 3)
np.tile(vec1, 3)

vec1 +vec1

max(vec1)
sum(vec1)

#35672 이하 홀수들의 합?
X = np.arange(0, 35672, 2)
sum(X)
X.sum()


b = np.array([1, 2, 3], [4, 5, 6])
len(b)

a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
a + b # 출력 오류

np.tile(a, 2) + b
np.repeat(a, 2) + b
b == 3

# 35672보다 작은 수 중 7로 나눠서 나머지가 3인 숫자들의 개수
sum((np.arange(1, 35672) % 7) == 3)


# 10보다 작은 수 중 7로 나눠서 나머지가 3인 숫자들의 개수
sum((np.arange(1, 10) % 7) == 3)





















