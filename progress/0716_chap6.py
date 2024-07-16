import numpy as np
import pandas as pd

df = pd.read_csv('./data/exam.csv')

# 데이터 전처리 함수
# query() : 조건에 맞는 행을 걸러내는 함수
df.query('nclass == 1')
df.query('math > 50')
df.query('math < 50')
df.query('english >= 50')
df.query('english <= 80')
df.query('nclass == 1 & math >=50')
df.query('nclass == 2  &  math >= 80')
df.query('english >= 90  |  math >= 90')
df.query('english < 90  |  science < 50')
df.query('nclass == 1  |  nclass == 3 | nclass == 5')
df.query('nclass in [1, 3, 5]')
df.query('nclass not in [1, 3, 5]') # not in 가능
df[~df["nclass"]]

# df[] 변수 추출
df['math']
df[['math', 'nclass']]
df.query('nclass == 1')[['math', 'english']]

df.query('nclass == 1')\
 [['math', 'english']]\
.head

# drop() 변수 제거
df.drop(columns = 'math')


# sort_values() 정렬하기
df.sort_values('math')
df.sort_values('math', ascending = False) # ascending = False 내림차순
df.sort_values(['math', 'nclass'], ascending = [True, False]) 

# assign() 변수 추가 , 여러 개 칼럼 추가 가능 ??? 
df = df.assign(
    total = df['math'] + df['english'] + df['science'],
    mean = (df['math'] + df['english'] + df['science']) / 3
    ).sort_values("total", ascending=False)
    
# 람다 이용 ???
df = df.assign(
    total = lamda x: x['math'] + x['english'] + x['science'],
    mean = lamda x: x["total"] / 3
    ).sort_values("total", ascending=False)

df.head

# groupby() 
# agg() 요약하다
df.agg(mean_math = ("math", "mean"))
df.groupby("nclass") \
            .agg(mean_math = ("math", "mean"))
            
# 예제1
df.groupby('nclass', as_index = False) \
    .agg(mean_math = ('math', 'mean'))

df.groupby('nclass')\
    .agg(mean_math = ('math', 'mean'),
        sum_math = ('math', 'sum'),
        median_math = ('math', 'median'),
        n = ('nclass', 'count')
    )
# 예제2
mpg = pd.read_csv("./data/mpg.csv")
mpg.head()
mpg.query('category == "suv"')\
    .assign(total = (mpg['hwy'] + mpg['cty']) / 2)\
    .groupby('manufacturer')\
    .agg(mean_tot = ('total', 'mean'))\
    .sort_values('mean_tot', ascending = False)\
    .head()
    
# merge()
# concat()















