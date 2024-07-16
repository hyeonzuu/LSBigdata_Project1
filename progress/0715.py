#브로드 캐스팅
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b
a.shape
b.shape

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
[10.0, 10.0, 10.0],
[20.0, 20.0, 20.0],
[30.0, 30.0, 30.0]])
matrix.shape # 4행 3열

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
result 

vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector
vector.shape
result = matrix +vector
result

# 슬라이싱
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(42)
a = np.random.randint(1, 21, 10) #rand +int, 1부터 20까지 10개를 랜덤으로 추출
a

a[2:5:step]
a[-1] # 끝에서 첫번째
a[-2]
a[::2] # 처음 부터 간격 2씩 출력
a[1:6:2]

# 1에서부터 1000사이의 3의 배수
sum(np.arange(1, 1000) % 3 == 0)
sum(np.arange(3, 1001, 3)) 
x = np.arange(3, 1001)
sum(x[::3])

# delete
np.delete(a, [1, 3])

# 논리 연산자
a > 3
b = a[a > 3]
print(b)

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a > 2000) & (a <5000)]
# a[조건을 만족하는 논리형 벡터]

!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
np_df= np.array(df['mpg']) # array로 하면 숫자만 출력
np_df

model_names = np.array(df.index)


# 15이상 25 이하 데이터 개수
sum((np_df >= 15) & (np_df <= 25))

# 평균 mpg보다 이상인 자동차 대수
sum(np_df >= np.mean(np_df))
# 평균 mpg 이상인 모델
model_names[sum(np_df >= np.mean(np_df))]

# 15작거나  25 이상 데이터 개수
sum((np_df < 15) | (np_df >= 25))

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B","C","F","W"])
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)] # ??? 모르겠음
(a > 2000) & (a < 5000)

a[a > 3000] = 3000
a 

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
a[a >5000][0]
np.where(a > 10000)[0][0]

# 처음으로 10000보다 큰 숫자 중 50번쨰로 나오는 숫자?
x = np.where(a > 10000)
x
y = x[0][49] # 81번째에 숫자가 있다
y

# 처음으로 500보다 작은 숫자와 위치?
x = np.where(a < 500)
a[x[0][-1]]


# 빈칸 나타내기
a = np.array([20, np.nan, 13, 24, 309])
a
a + 3
np.mean(a)
np.nanmean(a) # nan을 무시하고 평균값 계산
np.nan_to_num(a, nan = 0) # nan을 내가 원하는 값으로 변경
np.isnan(a)


False
a = None
b = np.nanb
a

b + a # None 수치연산 불가능

np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered


# 벡터 합치기
str_vec = np.array(['사과', '배', '수박', '참외'])
str_vec


mix_vec = np.array(['사과', 12, '수박', '참외'], dtype=str) # dtype=str 문자열로 받아라
mix_vec # 리스트는 섞는 거 허용


combined_vec = np.concatenate((str_vec, mix_vec)) # 입력값이 리스트, 튜플 =>  넘파이 어레이로 묶임
combined_vec # UC :Unicode String



col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5),
                                  np.arange(12, 18)))
uneven_stacked

# 길이가 다른 벡터
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 17)
vec2
vec1
vec1 = np.resize(vec1, len(vec2))
vec1

col_stacked = np.column_stack((vec1, vec2))
col_stacked

row_stacked = np.vstack((vec1, vec2))
row_stacked

# 연습문제
a = np.array([12, 21, 35, 48, 5])
a[0::2]

a.max()
np.unique(a)

a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(6, dtype=int)
c[::2] = a
c[1::2] = b
c

# 교재 4장
import numpy as np
import pandas as pd

df = pd.DataFrame({'name' :['김지훈', '이유진', '박동현', '김민지'],
                'english' : [90, 80, 60, 700],
                'math' :[50 ,60, 100, 20]
                })
df

type(df) # 판다스에서 정의한 데이타프레임
df["name"]
type(df["name"]) # series 
sum(df['english'])
np.mean(df['english'])
sum(df['english']) / 4

# 예제
df = pd.DataFrame({'제품' : ['사과', '딸기', '수박'],
                    '가격': [1800, 1500, 3000],
                    '판매량' : [24, 38, 13]})

np.mean(df['가격'])
np.mean(df['판매량'])
df

# 파일 불러오기
import pandas as pd
import numpy as np
df_exam = pd.read_excel('./data/excel_exam.xlsx') # header = None => 헤더 X
df_exam
df_exam.head()

sum(df_exam['english']) / len(df_exam)
sum(df_exam['math']) / len(df_exam)
sum(df_exam['science']) / len(df_exam)

df_exam.shape
len(df_exam)

df_exam = pd.read_excel('./data/excel_exam.xlsx', sheet_name = "sheet2") # 불러오고 싶은 시트 불러오기4


df_exam["Total"] = df_exam["math"] + df_exam["english"] +df_exam["math"]
df_exam["Mean" ] = (df_exam["Total"] / 3).astype(int)
df_exam.head()

df_exam[df_exam["math"] > 50] # 전체 행에 적용
df_exam[(df_exam["math"] > 50) & (df_exam['english'] > 50) & (df_exam["math"] > 50)] 

# 수학 평균이상 & 영어 평균 이하
mean_m = np.mean(df_exam['math'])
mean_e = np.mean(df_exam['english'])

df_exam[(df_exam["math"] >= mean_m) & (df_exam["english"] <= mean_e)]

# 3반 학생들 꺼내오기
df_nc3 = df_exam[df_exam["nclass"] == 3]
df_nc3[["math", "english", "science"]]
df_nc3[1:2]



a = np.array([4, 2, 5, 3, 6])
a[2]

df_exam[0:10:2]
df_exam[7:16]

df_exam.sort_values("nclass", ascending = False) # ascending = False 내림차순
df_exam.sort_values(["nclass", "math"], ascending = [True, False]) # ascending = False 내림차순

a > 3
np.where(a > 3, "Up", "Down") # nparray 형태로 반환
df_exam["UpDown"] = np.where(df_exam["math"] > 50, "Up", "Down") 
df_exam.head()
