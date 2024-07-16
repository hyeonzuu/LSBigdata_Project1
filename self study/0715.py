import numpy as np
import pandas as pd

np.random.seed(42)
a = np.random.randint(1, 21, 10)

a
b = a[a > 4]
b
b
# 논리 연산자와 조건문
b = a[(a > 2) & (a < 9)]
b

a[a == 8] 
a[a != 8]

b = a[a % 3 == 0]
b

a = np.array([1, 2, 3, 4, 16, 17, 18]) # 예시 배열
result = a[(a == 4) & (a > 15)]
print(result)


a = np.array([1,5,7,8,10])

result = np.where(a < 7)
result


import numpy as np

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)

# 22000보다 큰 숫자의 위치 찾기
x = np.where(a > 22000)
x
# 첫 번째 22000보다 큰 숫자의 위치
my_indx = x[0][0]
my_indx
# 첫 번째 22000보다 큰 숫자와 위치 출력
first_value_above_22000 = a[my_indx]
print("첫 번째 22000보다 큰 숫자의 위치:", my_indx)
print("첫 번째 22000보다 큰 숫자:", first_value_above_22000)

# where ??모르겠음
np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

# 처음으로 22000보다 큰 숫자와 위치?
x = np.where(a > 22000)
x
type(x)
my_indx = x[0][0] # x[0]의 첫 번쨰 숫자를 출력하기 위해
my_indx
x[my_index]
print("위치 : ", my_indx)
print("수 : ", a[10])



# 처음으로 10000보다 큰 숫자 중 50번쨰로 나오는 숫자?
x = np.where(a > 10000)
x
x_indx = x[0][49]
x_indx
a[x_indx]

# 처음으로 500보다 작은 숫자와 위치?
x = np.where(a < 500)
y = a[x[0][-1]]
y
a[y]

