# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()
#초기값을 가진 리스트
numbers = [1,2,3,4,5]
range_list = list(range(5))

range_list[3] = "LS 빅데이터 스쿨"

range_list[1] = ["1st", "2nd", "3rd"]
range_list[1][2] # '3rd' 출력

# 리스트 내포(comprehension)
# 1. [] => 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현 x range(10)
# 3. for.. in.. 을 사용해서 원소정보 제공
my_list = list(range(10))
squares = [x**2 for x in range(10)]

import numpy as np
import pandas as pd
# 3, 5, 2, 15의 3제곱
my_squares = [x**3 for x in [3, 5, 2, 15]]
my_squares = [x**3 for x in np.array([3,5,2,15])] #np.array 가능

exam = pd.read_csv("data/exam.csv")
exam["math"]
my_squares = [x**3 for x in exam["math"]] # 시리즈 가능 


"안녕" + "하세요"

# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 +list2

list1 * 3 + list2 * 5


# 리스트 각 원소별 반복
numbers = [5, 2, 3]
# 1. x for x in numbers에서 numbers에서 숫자를 하나씩 꺼내온다
# 2. for _ in range(3) 3(0, 1, 2)회 반복
repeated_list = [x for x in numbers for _ in range(3)]
repeated_list = [x for x in numbers for _ in [5, 4, 3, 2]] # 4번 반복
repeated_list = [x for x in numbers for y in range(3)]
for i in numbers:
    for j in range(4):
        print(i)
        
# 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]


# for 루프 문법
# for i in 범위 :
# 작동방식
for x in [4, 1, 2, 3]:
    print(x)

for x in range(5):
    print(x**2)

# 리스트를 하나 만들어서
# for 루프를 사용해서 2,4,6,8,..,20수를 채워 넣어보세요
[i for i in range(2, 21, 2)]

mylist = []
for i in range(1, 11):
    mylist.append(i*2)

mylist = [0] * 5

mylist = [0] *10
for i in range(10):
    mylist[i] = 2*(i + 1) 
    
# 인덱스 공유해서 카피하기    
# 퀴즈 :mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 8, 10, 12, 14, 35, 23, 20]
mylist = [0] * 5
for i in range(5):
    mylist[i] = mylist_b[2 * i]


for x in numbers:
    for y in range(4):
    print(x)

# _의 의미
# 앞에 나온 값을 가리킴
5 + 4
_ + 6 # 앞 식의 결과 값인 9가 대입

# 값 생략, placehoder
a, _, b, = (1, 2, 4) # _ = 2
a; b

_= None
del _

# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 []묶어줌 :리스트로 반환하기 위해서
# for 루프의 : 생략
# 실행부분을 먼저 써준다.
# 중복부분 제외
newlist = [i*2 for i in range(1, 11)]

for i in range(5):
    print("hello")

for i in range(5):
    print(i)

# 0을 고정시키고 j 숫자가 돌아간다음 i의 다음 숫자로 넘어감 (반복)
for i in range(3):
    for j in range(2):
        print(i, j)


for i in range(3):
    for j in range(2):
        print(i)
        
# 원소 체크
fruits = ["apple", "apple", "banana", "cherry"]
mylist = []
for x in fruits:
    mylist.append(x == "banana")
    
# 바나나의 위치를 출력 하려면?
my_index = 0
for x in fruits:
    mylist.append(x == "banana")
my_index

fruits = np.array(fruits) # fruits 자체가 리스트니까
int(np.where(fruits == "banana")[0][0]) # 튜플이니까 


# 원소 거꾸로 써주는 reverse
fruits.reverse()
# 원소 맨 끝에 붙여주기
fruits.append("pinapple")

# 원소 특정 위치에 추가
fruits.insert(2, "test")

#원소 제거
fruits.remove("test")

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])
# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove) # 바나나와 사과가 아닌 것들 True
# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]



