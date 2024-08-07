# 패키지 로드
import numpy as np
import pandas as pd

# 데이터 자료 만들기
tab3 = pd.read_csv("data/tab3.csv")

tab1 = pd.DataFrame({"id" : np.arange(1, 13),
                     "score" : tab3["score"]})
                     
tab2 = tab1.assign(gender=["female"]*7 + ["male"]*5)


# 1. 표본 t 검정 (그룹 1개)
# 귀무가설 vs. 대립가설
# H0: mu = 10 vs mu != 10
# 유의수준 5%로 설정

from scipy.stats import ttest_1samp
tab1
# tab1의 score만 들어가야함
result = t_statistic, p_value = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
print("t_statistic:", t_statistic)
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률
tab1["score"].mean() #표본평균
result.statistic # t 검정통계량
result.pvalue # 유의확률 (p-value)
result.df # 자유도

# 귀무가설이 참일 때, 11.53(표본평균)이 관찰될 확률이 6.48%이므로 
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05(유의수준)보다 크므로 귀무가설을 거짓이라 판단하기 힘들다. 
# 유의 수준보다 p-value가 높아서 기각  못한다(+귀무가설 채택). (0.05 < 0.064)
# 95% 신뢰구간
ci = result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1] # ci[0] < 신뢰구간 < ci[1]

# 2표본 t 검정 (그룹2)
# 분산 같은 경우 : 독립 2표본 t검정
# 분산 다른 경우 : 웰치스 t 검정

# 귀무가설 vs. 대립가설
# H0: mu_m = mu_f vs mu_m > mu_f
# 유의수준 1%로 설정
tab2
from scipy.stats import ttest_ind

male = tab2[tab2['gender'] == 'male']
female = tab2[tab2['gender'] == 'female']

result = ttest_ind(female["score"], male["score"],
                            alternative="less", equal_var=True) #alternative 대립가설 기준
# alternative="less"의 의미는 첫번째 입력그룹의 평균이 두 번째 입력그룹 평균보다 작다.
# female이 먼저 나왔으니까 less가 맞음

result.statistic # t 검정통계량
result.pvalue # 유의확률 (p-value)
result.df # 자유도

ci = result.confidence_interval(confidence_level=0.95)
ci[0] # 단측 검정이라서
ci[1] 

# 대응 표본 t검정 (짝지을 수 있는 표본)
# 귀무가설 vs. 대립가설
# H0: mu_bf = mu_af vs mu_bf > mu_af
# H0: mu_d = 0 vs mu_d > 0
# mu_d = mu_af - mu_bf
# 유의수준 1%로 설정
tab3_data = tab3.pivot_table(index='id',
                            columns='group',
                            values='score')
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

# 데이터 모양으로 1samp, ind 결정
result = ttest_1samp(test3_data, popmeam=0, alternative='greater')

result.statistic # t 검정통계량
result.pvalue # 유의확률 (p-value)
result.df # 자유도

ci = result.confidence_interval(confidence_level=0.95)
ci[0] 
ci[1] # 단측 검정이라서

-----------------------------------------------------------------------------
# tab3_data = tab3.pivot_table(index = 'id', columns = 'group', values = 'score').reset_index() # long to wide
# 
# tab3_data.melt(id_vars = 'id', value_var = ['A', 'B'], var_name = "group", value_name = 'score') # wide to long

# pivot 만들기 연습1
df = pd.DataFrame({"id" : [1, 2, 3],
                    "A" : [10, 20, 30],
                    "B" : [40, 50, 60]
                    })

df_long = df.melt(id_vars = 'id', 
        value_vars = ['A', 'B'],
        var_name = "group", 
        value_name = 'score')
# wide to long

df_long.pivot_table(
    columns = "group",
    values= "score",
    aggfunc = "max"
)


# 연습 2
import seaborn as sns
tips = sns.load_dataset("tips").reset_index(drop=False)

tips.pivot_table(columns='day', values='tip')

# 요일별로 펼치고 싶은 경우
tips.columns.delete(4)
tips.pivot_table(index = 'index',columns='day', values='tip').reset_index()
---------------------------------------------------------------------------------
























