import numpy as np
import pandas as pd

exam = pd.read_csv('./data/exam.csv')
# chap 7 데이터 정제
df = pd.DataFrame({'sex' : ['M', 'F', np.nan, "M", "F"],
                    'score' : [5, 4, 3, 4, np.nan]})
df
pd.isna(df) # na인 곳에만 True

df["score"] + 1

pd.isna(df).sum()

# 결측치 제거하기
df.dropna(subset = 'score') # score 변수에서 결측치 제거
df.dropna(subset = ['score', 'sex']) # 여러변수 결측지 제거법
df.dropna()

# 데이터 프레임 location을 통한 인덱싱 ???
exam.loc[행인덱스, 열 인덱스 ]
exam.loc[[2, 7, 14], ]
exam.loc[:3]
exam.iloc[:3]
exam.iloc[0,0]
exam.loc[[0],['id', 'nclass']]
exam.iloc[[2,7,4], 3] = np.nan
exam.loc[[2,7,4], ["math"]] = np.nan
exam

# 수학점수가 50점 이하인 학생 점수 50으로 상향조정
exam.loc[exam["math"] <= 50, ["math"]] = 50
exam

# 영어점수 90점 이상 90으로 하향 조정
# iloc은 숫자 벡터가 나와야 조회 가능 ??? 선생님 코드 참고
exam.loc[exam["english"] >= 90, "english"] 
exam.iloc[exam["english"] >= 90, 3] = 90
exam

# math  점수 50점 이하 -로 변경
exam = pd.read_csv('./data/exam.csv')
exam.iloc[exam["math"] <= 50, 2] = '-'

# - 를 수학점수 평균으로 대체
# 1
exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean
# 2
math_mean = exam.query('math not in ["-"]')["math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean
# 3
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean
exam
# 4
exam.loc[exam["math"] == '-', ["math"]] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam["math"]), ["math"]] = math_mean

# 5 굳이 이해 안해도 됨
vector = np.array([np.nan if x == '-' else float(x) for x in exam["math"]]).mena()
vector = np.array([float(x) if x != '-' else np.nan for x in exam["math"]])
vector
exam["math"] = np.where(exam["math"] == "-", math_maen, exam["math"])

df.loc[df["score"] == 3.0, ["score"]] = 4
df



























