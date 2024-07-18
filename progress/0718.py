# 행렬 
import numpy as np
import matplotlib.pyplot as plt

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.vstack(
                        (np.arange(1, 5),
                         np.arange(12, 16)))

print("행렬: \n", matrix)


np.zeros(5)
np.zeros((5,4))

np.arange(1, 5).reshape([2,2])
np.arange(1, 7).reshape([2,3])
np.arange(1, 7).reshape([2,-1]) # -1 통해서 크기를 자동으로 결정

# Q1 0~99까지 수 중 랜덤하게 50개 숫자를 뽑아서 5,10 행렬 만들기
np.random.randint(0, 100, 50).reshape(5, -1)

a = np.arange(1, 21).reshape((4, -1), order = "F")

# 행렬의 원소에 접근하기
# 인덱싱
a[0, 0]
a[0:2, 3]
a[2, 3]
a[1:3, 1:4]
# 네번쨰 행만 가져와, 열부분이 비어있음 => 전체 열 출력
a[3, ]
a[3, :]
a[3, ::2]

# 짝수행만 출력
b = np.arange(1, 101).reshape((20, -1))
b[1::2, ]
b[[1, 4, 6, 14], ]

# 조건문을 활용한 필터링
x = np.arange(1, 11).reshape((5,2)) * 2
x[[True, True, False, False, True],0]

b[: , 1: 2] # 메트릭스(행렬)
b[: , (1,)] # 메트릭스
b[: , 1] # 자동으로 차원 축소(벡터 :1차원)
b[: , 1] .reshape((-1, 1)) # 메트릭스

# 조건문으로 필터링
b[b[: , 1] % 7 == 0, :] # 행렬로 출력
b[: , 1] % 7 == 0 # 조건문(True, False로 출력)


# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
plt.clf()

a = np.random.randint(0, 10, 20).reshape(4, -1)
a/9

import urllib.request
import imageio


img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

# 이미지 읽기
jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

plt.show(jelly)
plt.show(jelly[:,:,0].transpose())
plt.show(jelly[:,:,0])
plt.show(jelly[:,:,1])
plt.show(jelly[:,:,2])
plt.show(jelly[:,:,3])
plt.axis('off')
plt.show()
plt.clf()

# 배열 다루기
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

# 3차원 배열로 합치기
array1 = np.array([mat1, mat2])
array1.shape

first_slice = array1[0, :, : ]
filtered_array = array1[:, :, :, -1]

# 예제
array1[:, :, [0, 2]]
array1[:, 0, :]
array1[0, 1, [1, 2]]
# 위아래 == 
array1[0, 1, 1:3]
x = np.arange(1, 101).reshape((5, 5, -1))
x = np.arange(1, 101).reshape((-1, 5, 2))


array2 = np.array([array1, array1])
array2.shape

# 넘파이 배열 메서드
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()
a.sum(axis=0) # 열 덧셈
a.sum(axis=1) # 행 덧셈


a.mean()
a.mean(axis=0) # 열 덧셈
a.mean(axis=1) # 행 덧셈

mat_b=np.random.randint(0, 100, 50).reshape((5, -1))
mat_b

# 가장 큰 수는?
mat_b.max()

# 행별 가장 큰수는?
mat_b.max(axis=1)

# 열별 가장 큰수는?
mat_b.max(axis=0)

a=np.array([1, 3, 2, 5])
a.cumsum()

a=np.array([1, 3, 2, 5])
a.cumprod()

mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)

mat_b.reshape((2, 5, 5)).flatten()
mat_b.flatten()


d = np.array([1, 2, 3, 4, 5])
d.clip(2, 4)

d_list=d.tolist()
d
d_list
------------------------------------------------------------------------------
# 균일 확률 변수 만들기
np.random.rand(1)

def X(num):
    return np.random.rand(num)

X(3) # 한번에 세 개가 뽑힌게 아니라 순차적으로 세개를 뽑아서 한 번에 출력
X(1)

# 베르누이 확률 변수 모수: P
def Y(p, num):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0) 
Y(0.5, 100).mean() 
# 대수의 법칙 : num이 많을 수록 표본비율과 실제비율이 비슷해진다

# 새로운 확률변수
# 가질 수 있는 값 :0, 1, 2
#               20% 50% 30%
def Z():
    x = np.random.rand(1)
    return np.where(x < 0.2, 0,
                    np.where(x < 0.7, 1, 2)) 
Z()

def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0,
                    np.where(x < p_cumsum[1], 1, 2)) 
                    
p = np.array([0.2, 0.5, 0.3])
Z(p)





