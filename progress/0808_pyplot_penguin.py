import plotly.express as px
!pip install palmerpenguins
import pandas as pd
import numpy as np
from palmerpenguins import  load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(penguins, x='bill_length_mm', y='bill_depth_mm', 
                color="species", size_max=15) # trendline = "ols" 회귀직선 그려줌

# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white", size=18)), 
        tickfont=dict(color="white", size=14),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white", size=18)), 
        tickfont=dict(color="white", size=14),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(title=dict(text="펭귄종", font=dict(color="white", size=18)), font=dict(color="white"))
)

# 점 크기 키우기
fig.update_traces(marker=dict(size=10), opacity = 0.7)

fig.show()
# 선형회귀 모델
from sklearn.linear_model import LinearRegression
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins=penguins.dropna()

penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)

penguins_dummies.columns
penguins.columns
penguins_dummies.iloc[:,-3:]
# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

a = model.coef_
b = model.intercept_ 

regline_y = model.predict(x)
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                 legend=False)
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,
                color="black")
plt.show()
plt.clf()

# y = 0.2 * bill_length -1.93 * species_Chins!pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins.info()
penguins["species"].unique()
penguins.columns

# x: bill_length_mm
# y: bill_depth_mm

fig=px.scatter(penguins, x= "bill_length_mm", y= "bill_depth_mm", color="species", trendline="ols")
# dict() = {}
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이와 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        font=dict(color="white", size=14),  # 범례 폰트 크기 조정
        title=dict(text="펭귄 종", font=dict(color="white", size=14))  # 범례 제목 조정
    )
)
fig.update_traces(marker=dict(size=12, opacity=0.7)) 
fig.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

# 심슨의 역설(숨어있는 변수에 따라서 트랜드가 바뀜(분석이 바뀌게 됨))
model.fit(x, y)
linear_fit=model.predict(x)
model.coef_ # 부리 길이가 1mm가 증가할때마다 부리 깊이가 0.08씩 줄어든다
model.intercept_

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
# 3개 중 2개만 있어도 정보를 다 얻을 수 있음(drop_first = True)
penguins=penguins.dropna()
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

model = LinearRegression()
model.fit(x, y)

model.coef_ 
model.intercept_

# 회귀직선
# y = 0.2 * bill_length - 1.93 * species_Chinstrap - 5.1 * species_Gentoo + 10.56
#       species     island  bill_length_mm  ...  body_mass_g     sex  year
#1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
#340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
# x1, x2, x3
# 39.5, 0, 0
# 43.5, 1, 0
# 0.2 * 43.5 +- 1.93 * True(1) -5.1 * False + 10.56

regline_y=model.predict(x)

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                legend=False)
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,
                color="black")
plt.show()
plt.clf()

trap -5.1 * species_Gentoo + 10.56
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
0.2 * 40.5 -1.93 * True -5.1* False + 10.56


fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)

fig.show()






