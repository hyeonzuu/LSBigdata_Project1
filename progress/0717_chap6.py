# chap 6.7 데이터 합치기
import numpy as np
import pandas as pd

exam = pd.read_csv('./data/exam.csv')
# 가로로 데이터 합치기
# 중간고사 데이터 만들기
test1 = pd.DataFrame({
    'id' : [1, 2, 3, 4, 5],
    'midtern' :[60, 80, 70, 90, 85]})
test2 = pd.DataFrame({
    'id' : [1, 2, 3, 4, 5],
    'final' :[70, 83, 65, 95, 80]})

test1
test2

# left join 중요
total = pd.merge(test1, test2, how = 'left', on = 'id')
# how = 왼쪽으로 붙여, on = id를 기준으로 

#right join
total = pd.merge(test1, test2, how = 'right', on = 'id')
total


test1 = pd.DataFrame({
    'id' : [1, 2, 3, 4, 5],
    'midtern' :[60, 80, 70, 90, 85]})
test2 = pd.DataFrame({
    'id' : [1, 2, 3, 40, 5],
    'final' :[70, 83, 65, 95, 80]})
    
total = pd.merge(test1, test2, how = 'left', on = 'id')
total # 40이 출력이 안됨 => 왼쪽(midtern)을 기준으로 결합했기 때문

# Inner join : 공동으로 있는 id만 출력 (교집합)
total = pd.merge(test1, test2, how = 'inner', on = 'id')
total

# outer join : 있는 id 모두 출력 (합집합)
total = pd.merge(test1, test2, how = 'outer', on = 'id')
total

name = pd.DataFrame({'nclass' : [1, 2, 3, 4, 5],
                    'teacher' :['kim', 'lee', 'park', 'choi', 'jung']})
name
df
df_name = pd.merge(df, name, how = 'left', on = 'nclass')
df_name

# 세로로 데이터 합치기
score1 = pd.DataFrame({
    'id' : [1, 2, 3, 4, 5],
    'score' :[60, 80, 70, 90, 85]})
score2 = pd.DataFrame({
    'id' : [6, 7, 8, 9, 10],
    'score' :[70, 83, 65, 95, 80]})
score1
score2

score_all = pd.concat([score1, score2])
score_all
