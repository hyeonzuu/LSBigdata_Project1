import numpy as np
import pandas as pd

df = pd.read_csv('./data/exam.csv')
df.head()
df.tail()
df.tail(10) # 뒤에 10개 출력
df.shape # ()없이
# 메서드(함수) vs 속성/어트리뷰트(변수)
# 함수 () O, 어트리뷰트는 () X 

df.info() # columns 개수, 데이터 타입, 변수 값의 개수 
df.describe() # count, mean, 표준편차(std) 등 
df.columns

# 메서드 
df # df : 판다스 데이터 프레임 
type(df)
var = [1, 2, 3] # 리스트
type(var) # head 메서드 없어서 출력 오류

# 변수명 바꾸기
df2 = df.copy()
df2
df2.rename(columns={'nclass' : 'class'}) 
df2 = df2.rename(columns={'nclass' : 'class'}) # 업데이트를 해줘야함
df2.info()

# 변수 파생
df2['total'] = df2['math'] + df2['english'] + df2['science']
df2.head()

# 합불합 변수 파생
df2['grade'] = np.where(df2['total'] >= 200, 'Pass', ' Fail' )
df2.head()
df2.info()

df2['grade'].value_counts() # 값 개수 세기

# 막대그래프로 빈도
import matplotlib.pyplot as plt

grade = df2['grade'].value_counts()
grade.plot.bar(rot = 0) # rot = 0 x축 이름 회전
# == 둘다 똑같
df2['grade'].value_counts().plot.bar()

plt.show()
plt.clf()

# 범주 만들기
df2['grade2'] = np.where(df2['total'] >= 200, 'A',
                        np.where(df2['total'] >= 150,'B', 'C' ))

df2['grade2'].isin(["A" , "C"])                     
                        
df2.head()
grade = df2['grade2'].value_counts()
grade.plot.bar(rot = 0)
plt.show()
plt.clf()


# 복습! random을 중복없이 출력 (강사님 코드 참조)
a = np.random.choice(np.arange(1, 21), 10 ,replace=False)
a






















