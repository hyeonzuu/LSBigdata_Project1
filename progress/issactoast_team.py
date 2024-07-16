import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("./data/2018년월별국내여행횟수.xlsx")
df.head()
df.info()

# 1. 변수 이름 변경했는지?
df = df.rename(columns = {'15~19세' : '청소년'})
df = df.rename(columns = {"소계" : "total", "년":"year", "월":"month"})
df.info()

# 2. 행들을 필터링 했는지?
#2018년 남성의 여행 횟수가 평균보다 많은 달은 upper, 적은달은 lower로 대응하는 파생변수 man_count 생성
df['man_count'] = np.where( df['남자'] > df['남자'].mean(), "upper", "lower")
df

# 3. 새로운 변수를 생성했는지?
df = df.assign(
    청년층 = df['20대'] + df['30대'],
    중년층 = df['40대'] + df['50대']
)
df.columns
df.head()

# 4. 그룹 변수 기준으로 요약을 했는지?
df.groupby('month').agg(청년층_mean =('청년층', 'mean')).sort_values('청년층_mean', ascending = False)

# 5. 정렬 했는지
df.sort_values(["total"], ascending = [ False])

df.to_csv('df_pre.csv', index=False)


