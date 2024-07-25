import numpy as np


# E[x]
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)

# 베르누이
def B(p):
    x = np.random.rand(1)
    return np.where(x > p, 1, 0)
B(0.5)


---------------------------------------------------------------------------------
# stat2
import matplotlib.pyplot as plt

data = np.random.rand(10)
plt.hist(data, bins = 30, alpha = 0.7, color = 'blue')
plt.title("Histogram")
plt.show()
plt.clf()

# 연습문제 
x = np.random.rand(10000, 5).mean(axis=1)

x = np.random.rand(50000).reshape(-1, 5).mean(axis=1)
plt.hist(x, bins = 30, alpha = 0.7, color = 'blue')
plt.title("Histogram")
plt.show()
plt.clf()



