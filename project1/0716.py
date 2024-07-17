import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/LSBigdata_Project1/pre_select.xlsx")
df.head()

# 년도별로 슬라이싱
df_2018 = df.iloc[:15]
df_2019 = df.iloc[15:30]
df_2020 = df.iloc[30:45]
df_2021 = df.iloc[45:60]
df_2022 = df.iloc[60:]

df.describe()
# 
