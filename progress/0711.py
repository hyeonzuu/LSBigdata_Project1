# 패키지 활용하기
import seaborn as sns
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var

sns.countplot(x = var, palette="viridis")
plt.show()
plt.clf()

df = sns.load_dataset('titanic')
sns.countplot(data = df, x= 'sex', hue = "sex")
plt.show()
plt.clf()

sns.countplot(data = df, x = 'class')
plt.show()
plt.clf()

sns.countplot(data = df, y = 'class', hue = 'sex')
plt.show()

#!pip install scikit-learn
from sklearn import metrics 
metrics.accuracy_score()

