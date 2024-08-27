import numpy as np
import pandas as pd
from scipy.stats import uniform, norm
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# 초기 설정
example_list = list(range(0, 30))
np.random.shuffle(example_list)
groups = [example_list[6 * i : 6 * i + 6] for i in range(0, 6)]

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y": y,
    "x": x
})

# 다양한 alpha 값을 위한 리스트
alphas = np.arange(0.1, 0.5, 0.001)

# 폴드별 예측 결과를 저장할 리스트
all_predictions = []

# 각 alpha에 대한 성능 결과를 저장할 리스트
performance_results = []

# 폴드 순회 및 cross-validation 진행
for alpha in alphas:
    fold_mse = []
    
    for fold in range(5):
        # 검증 세트와 훈련 세트 구분
        valid_idx = groups[fold]
        train_idx = []
        for group in groups:
            if group != valid_idx:
                train_idx.extend(group)

        train_df = df.loc[train_idx].copy()
        valid_df = df.loc[valid_idx].copy()

        # 다항식 특성 생성
        for i in range(2, 21):
            train_df[f"x{i}"] = train_df["x"] ** i
            valid_df[f"x{i}"] = valid_df["x"] ** i

        # 훈련 데이터와 검증 데이터 분리
        train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        train_y = train_df["y"]

        valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        valid_y = valid_df["y"]

        # Lasso 모델 학습
        model = Lasso(alpha=alpha)
        model.fit(train_x, train_y)

        # 검증 데이터에 대한 성능 계산 (MSE)
        mse = np.mean((model.predict(valid_x) - valid_y) ** 2)
        fold_mse.append(mse)


    # 각 alpha에 대한 평균 MSE 계산
    mean_mse = np.mean(fold_mse)
    performance_results.append(mean_mse)

# 최적의 alpha 값 찾기
optimal_alpha = alphas[np.argmin(performance_results)]
print(f"Optimal Alpha: {optimal_alpha}")

# 성능 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(alphas, performance_results, marker='o')
plt.title("Lasso Regression - Validation MSE Across Alphas")
plt.xlabel("Alpha")
plt.ylabel("Validation MSE")
plt.show()
