import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('exam.csv')
# 앞에서 5행 출력
df.head()

##앞에서 10행까지 출력
df.head(10)

# 뒤에서 5행 출력
df.tail()

#데이터 구성
df.shape

# 데이터 정보
df.info()

# 데이터 요약
df.describe()

# 함수와 메서드 차이
# 내장함수
sum(var)

#패키지 함수
pd.read_csv()

#메서드
df.head() #변수가 지닌 함수

#어트리뷰트
df.shape #변수가 지니고 있는 값

# 변수명 바꾸기
df_raw = pd.DataFrame({'var1' : [1, 2, 1],
                        'var2' : [2, 3, 2]})
df_raw

# 데이터 프레임 복사본
df_new = df_raw.copy()
df_new

# 변수명 바꾸기
df_new = df_new.rename(columns = {'var2' : 'v2'})
df_new

# 파생 변수 만들기
df['total'] = df['math'] + df['english'] + df['science']
df.head()

df['mean'] = df['math'] + df['english'] + df['science'] / 3
df.head()



df['total'].describe()

#합격판별 변수 만들기
import numpy as np
df['test'] = np.where(df['total'] >= 200, 'pass', 'fail')
df.head()

df['test'].value_counts()

count_test = df['test'].value_counts() 
count_test.plot.bar()
plt.show()

#중첩 조건문
df['grade'] = np.where(df['total'] >= 250, 'A',
              np.where(df['total'] >= 200, 'B', 'C'))
df.head()

count_grade = df['grade'].value_counts()
count_grade

count_grade = df['grade'].value_counts().sort_index()
count_grade
 

