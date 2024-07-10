a = 1
a

# . 현재폴더
# .. 상위폴더
# show folder in new window : 해당위치 탐색기

# ps(powershell) 명령어 목록
# LS : 파일 목록
# cd : 폴더 이동
# cd .. : 상위폴더 이동
# cd ..\.. : 두 계단 위로 이동
# 첫 글자 누르고 Tab 하면 자동완성
# Tab+shift 자동완성
# cls : clear 

a = 10
a
# 변수 안에 문자열 넣기
a = '안녕하세요'
a

# 리스트 (변수 안에 여러개의 값 넣기)
a = [1, 2, 3]
a
b = [4, 5, 6]
c = a + b  # 출력 [1, 2, 3, 4, 5, 6]

# 문자열 리스트
a = "안녕하세요"
b = "LS 빅데이터 스쿨"
c = a + ' ' + b
c

num1 = 3
num2 = 5
print(num1)

# 기본 산술 연산자
a = 10
b = 3.3

print("a + b =", a + b) # 덧셈
print("a - b =", a - b) # 뺼셈
print("a * b =", a * b) # 곱셈
print("a / b =", a / b) # 나눗셈
print("a % b =", a % b) # 나머지
print("a // b =", a // b) # 몫
print("a ** b =", a ** b) # a듭제곱

# shift + Alt + 아래 화살표 => 아래로 복사
# ctrl + Alt + 아래 화살표 => 커서 여러개 

#비교 연산자
a == b
a != b
a < b
a > b
a <= b
a >= b
 
a = ((2**4) + (12453 //7)) % 8
b = ((9**7)/12) * (36452 % 253)
a < b

user_age = 25
is_adult = user_age >= 18
print("성인입니까?", is_adult)

a = "True"
b = TRUE 
c = true
d = True

TRUE = 3
b = TRUE

# True: 1
# False: 0
True + True
True + False
True - True

# True, False
a = True
b = False

a and b
a or b
not a

# and 연산자
True and False
True and True
False and True
False and False

# and로 치환 가능
True  * False
True  * True
False *  True
False *  False

# or 연산자
True  or False
True  or True
False or  True
False or  False

a = False
b = False
a or b
min(a+b, 1)

# 복합 연산자
a =3
a += 10
a
a -= 4
a
a %= 3
a

a += 12 
a **= 2
a

str1 = "hello"
str1 + str1
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

# 단항 연산자
x = 5
~5
# binary => 결과값 문자, '0b--' 2진수를 나타내는 표시
bin(-4)
bin(~0)
~-0

import pydataset
pydataset.data() # 입력값이 있어도 되고 없어도 됨. 데이터 불러와주는 함수.

df = pydataset.data("AirPassengers")
df

import pandas as pd
import numpy as np








