import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


raw_welfare = pd.read_spss("data/koweps_hpwc14_2019_beta2.sav")

# 복사본 만들기
welfare = raw_welfare.copy()

welfare.shape
welfare.info()
welfare.describe()

# rename
welfare = welfare.rename(
    columns={
            "h14_g3": "sex",
            "h14_g4": "birth",
            "h14_g10": "marriage_type",
            "h14_g11": "religion",
            "p1402_8aq1": "income",
            "h14_eco9": "code_job",
            "h14_reg7": "code_region"})

welfare = welfare[["sex", "birth", "marriage_type", "religion", "income", "code_job", "code_region"]]

welfare.shape

welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1, "male", "female")
welfare["sex"].value_counts()


welfare["income"].dtypes
welfare["income"].describe()
welfare["income"].isna().sum()

welfare["income"] > 9998

sex_income = welfare.dropna(subset = "income") \
                    .groupby("sex", as_index = False) \
                    .agg(mean_income = ("income", "mean"))


sns.barplot(data = sex_income, x = "sex", y = "mean_income", hue = "sex")
plt.show()
plt.clf()

# 숙제: 위그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 위아래 검정색 막대기로 표
# 95% 신뢰구간 계산
z = norm.ppf(0.95, loc=0, scale=1)
z
x.mean() + z * 6 / np.sqrt(16)
x.mean() - z * 6 / np.sqrt(16)


