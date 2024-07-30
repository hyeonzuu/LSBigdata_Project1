# 교재 8장 212p
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


economics = pd.read_csv("./data/economics.csv")

economics.info()
sns.lineplot(data = economics, x = 'date', y = 'unemploy')
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics["date"])
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.month_name()
economics["date2"].dt.day
economics["date2"].dt.quarter
economics["quarter"] = economics["date2"].dt.quarter

# 각 날짜가 무슨 요일인가?
economics["date2"].dt.day_name()
economics["date2"] + pd.DateOffset(days=3) 
economics["date2"] + pd.DateOffset(months=1) 

# 연도 변수 만들기
economics["year"] = economics["date2"].dt.year
economics.head()

# x 축에 연도 표시
sns.lineplot(data = economics, x = "year", y = "unemploy")
plt.show()
plt.clf()

# 신뢰구간 제거 ???모르겠다
sns.lineplot(data = economics, x = "year", y = "unemploy", errorbar = None)
plt.show()
plt.clf()
# sns.scatterplot(data=economics, x='year', y='unemploy', s=2)

my_df = economics.groupby('year', as_index=False)\
                 .agg(
                         mon_mean = ('unemploy', 'mean'),
                         mon_std = ('unemploy', 'std'),
                         mon_n = ('unemploy', 'count')
                      )
my_df

mean + 1.96 * std / sqrt(12)
my_df['left_ci'] = my_df['mon_mean'] - 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df['right_ci'] = my_df['mon_mean'] + 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])


x = my_df['year']
y = my_df['mon_mean']
plt.plot(x,y,color='black')
plt.scatter(x,my_df['left_ci'], color = 'green', s=5)
plt.scatter(x,my_df['right_ci'], color = 'green', s=5)
plt.show()
plt.clf()
























