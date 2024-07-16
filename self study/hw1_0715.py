 # do it 4장 84p 혼자서 해보기
 # Q1 다음 표의 내용을 데이터 프레임으로 만들어 출력해 보세요.
 import pandas as pd
 df = pd.DataFrame({'제품' : ['사과', '딸기', '수박'],
                    '가격' : [1800, 1500, 3000],
                    '판매량' : [24, 38, 13]})
df

import numpy as np
가격_평균 = np.mean(df['가격'])
판매량_평균 = np.mean(df['판매량'])
가격_평균
판매량_평균

# 5장 115p 혼자서 해보기
# Q1 mpg 데이터를 불러와 복사본을 만드세요
df = pd.read_csv('./data/mpg.csv')
df_new = df.copy()

# Q2 복사본 데이터를 이용해 cty는 city로 hwy는 highway로 바꾸세요
df_new = df_new.rename(columns = {'cty' :'city'}) 
df_new = df_new.rename(columns = {'hwy' :'highway'}) 

# Q3 데이터 일부를 출력해 변수명이 바뀌었는지 확인해보세요
df_new.head()

#  5장 130p 분석 도전
# Q1 데이터를 불러와 데이터 특징 파악
df = pd.read_csv('./data/midwest.csv')
df
df['poptotal']

# Q2 poptotal(전체 인구)변수를 total로 popasian(아시아 인구) 변수를 asian으로 수정하세요
df = df.rename(columns = {'poptotal' : 'total'})
df = df.rename(columns = {'popasian' : 'asian'})
df.head()
df.columns

# Q3 total, asian 변수를 이용해 '전체 인구 대비 아시아 인구 백분율' 파생변수를 추가하는 히스토그램을 만들어 분포를 살펴보세ㅇㅅ
df['asian%'] = df['asian'] / df['total'] * 100
import matplotlib.pyplot as plt
df['asian%'].plot.hist()
plt.show()

# Q4 아시아 인구 백분율 전체 평균을 구하고 평균을 초과하면 'large', 그외에는 'small'을 부여한 파생변수를 만드세요
asian_mean = df['asian%'].mean()
df['grade'] = np.where(df['asian%'] > asian_mean, 'large', 'small')
df

# Q5 large와 small에 해당하는 지역이 얼마나 많은지 빈도표와 빈도 막대 그래프를 만들어 확인해 보세요
count_grade = df['grade'].value_counts()
count_grade
count_grade.plot.bar(rot = 0)
plt.show()



