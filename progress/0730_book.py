import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'



welfare = pd.read_spss("data/koweps_hpwc14_2019_beta2.sav")

raw_welfare = pd.DataFrame(welfare)

# 복사본 만들기
welfare = raw_welfare.copy()

welfare.shape
welfare.info()
welfare.describe()

# rename
welfare = welfare.rename(
    columns={"h14_g3": "sex",
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
welfare["sex"].value_counts() # nall값 집계 안됨


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

welfare["birth"].dtypes
welfare["birth"].describe()
sns.histplot(data = welfare, x = "birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x ="age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income") \
                    .groupby("age", as_index = False) \ # as_index =  False  컬럼에 age 안들어감감
                    .agg(mean_income = ("income", "mean"))

sns.lineplot(data = age_income, x = "age", y = "mean_income")
plt.show()
plt.clf()


# 나이별 income 칼럼에서 na 갯수 세기
welfare["income"].isna().sum()
my_df = welfare.assign(income_na=welfare["income"].isna())\
                    .groupby("age", as_index = False) \
                    .agg(n = ("income_na", "count"))


sns.barplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

# 연령대에 따른 월급 차이
welfare["age"].head()
# 연령대 변수 만들기
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"] <= 59, "middle", "old")))
                                
# 빈도 구하기
welfare["ageg"].value_counts()

# 그래프 그리기
sns.countplot(data = welfare, x = "ageg", hue = "ageg")
plt.show()
plt.clf()

ageg_income = welfare.dropna(subset = "income") \
                     .groupby("ageg", as_index = False) \
                     .agg(mean_income = ("income", "mean"))

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income", hue = "ageg")
plt.show()
plt.clf()

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income", hue = "ageg", 
            order = ["young", "middle", "old"])
plt.show()
plt.clf()

# cut 함수수수퍼노바 
# 카테고리화 시키는 작업업
vec_x = np.random.randint(0, 100, 20)
pd.cut(vec_x, 10) # vec_x를 10개 구간으로 쪼개기
vec_y = np.arange(0, 10)
pd.cut(vec_y, 3)
# [(-0.009, 3.0] < (3.0, 6.0] < (6.0, 9.0]] 이렇게 3구간으로 쪼갠다

# 나이 0~9, 10~19, 20~29 세대별로 묶기
welfare['age_group'] = pd.cut(welfare['age'],
                         bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, np.inf], # 범위 나누기
                         labels=['baby', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대', 'old'], # 이름 붙이기
                         right=False)
welfare['age_group'].isna().sum()

age_income = welfare.dropna(subset = "income") \
                     .groupby("age_group", as_index = False) \
                     .agg(mean_income = ("income", "mean"))
sns.barplot(data = age_income, x = "age_group", y = "mean_income") 
plt.show()
plt.clf()


# 9-5 연령대 및 성별 월급 차이
# 연령대 및 성별 평균표 만들기
welfare["age_group"] = welfare["age_group"].astype("object") 
sex_income = welfare.dropna(subset="income")\
                    .groupby(["age_group", "sex"], as_index = False) \ # age_group 먼저 입력한 값 기준으로 묶임임
                    .agg(mean_income = ("income", "mean"))
                    
sns.barplot(data=sex_income, x = "age_group", y = "mean_income", hue = "sex")
plt.show()
plt.clf()

# 연령대별, 성별 4% 수입 찾기
x = np.arange(10)
np.quantile(x, q = 0.05)

welfare["age_group"] = welfare["age_group"].astype("object")
sex_income = welfare.dropna(subset="income")\
                    .groupby(["age_group", "sex"], as_index = False) \
                    .agg(mean_income = ("income", lambda x: np.quantile(x, q=0.96)) # lambda x =  income
                
sex_income

