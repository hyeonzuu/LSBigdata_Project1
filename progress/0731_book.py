# 9-6장
# merge 복습
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'



welfare = pd.read_spss("data/koweps_hpwc14_2019_beta2.sav")

raw_welfare = pd.DataFrame(welfare)

# 복사본 만들기
welfare = raw_welfare.copy()

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

welfare["code_job"]
welfare["code_job"].dtypes

list_job = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/LSBigdata_Project1/data/Koweps_Codebook_2019.xlsx", sheet_name="직종코드")
list_job.head()

welfare = welfare.merge(list_job, how="left", on="code_job")
welfare.dropna(subset="code_job")[["code_job", "job"]].head()

job_income = welfare.dropna(subset = ["job", "income"]) \
                    .groupby("job", as_index = False) \
                    .agg(mean_income = ("income", "mean"))
job_income.head()

# 수입 top10
top10 = job_income.sort_values("mean_income", ascending = False).head(10)

#막대 그래프 그리기
sns.barplot(data = top10, y = "job", x = "mean_income", hue = "job",)
plt.xticks(fontsize=8)
plt.yticks(fontsize=5)
plt.show()
plt.clf()

# 수입 bottom10
bottom10 = job_income.sort_values("mean_income").head(10)

#막대 그래프 그리기
sns.barplot(data = bottom10, y = "job", x = "mean_income", hue = "job",)
plt.xticks(fontsize=8)
plt.yticks(fontsize=5)
plt.show()
plt.clf()

# 9-7
job_male = welfare.dropna(subset = 'job') \
                  .query('sex == "male"') \
                  .groupby('job', as_index = False) \
                  .agg(n = ('job', 'count')) \
                  .sort_values('n', ascending = False) \
                  .head(10)

job_female = welfare.dropna(subset = 'job') \
                    .query('sex == "female"') \
                    .groupby('job', as_index = False) \
                    .agg(n = ('job', 'count')) \
                    .sort_values('n', ascending = False) \
                    .head(10)

# 9-8
welfare.info()
welfare["marriage_type"]
df = welfare.query("sex != 'marriage_type'") \
            .groupby("religion", as_index = False) \
            ["marriage_type"] \
            .value_counts(normalize=True) # 핵심  normalize=True하면 카운트가 아닌 비율로 들어감
            


df = welfare.query("marriage_type == 1") \
            .assign(proportion=df["proportion"]*100).round(1)
























