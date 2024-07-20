import numpy as np
import matplotlib.pyplot as plt
import math

def S_function(n, m, x0, sigma):
    n = - ( (100 * n / m - x0) ** 2 / (2 * sigma ** 2) )
    result = (1 / (math.sqrt(2 * math.pi) * sigma )) * np.exp(n)
    return result

# 設定參數
x0 = 67.45
sigma = 25.75
m = 1000

# 生成n的範圍
n_values = np.arange(1, 1001)

# 計算S_function的值
s_values = np.array([S_function(n, m, x0, sigma) for n in n_values])

# 計算總和
total_sum = np.sum(s_values)

# 找到最大值和最小值
max_value = np.max(s_values)
min_value = np.min(s_values)

# 輸出結果
print(f"Total sum of S_function values: {total_sum:.4f}")
print(f"Maximum value of S_function: {max_value:.4f}")
print(f"Minimum value of S_function: {min_value:.4f}")

# 繪製圖形
plt.figure(figsize=(10, 6))
plt.plot(n_values, s_values, label='S_function(n)')
plt.xlabel('n')
plt.ylabel('S_function(n)')
plt.title('S Function Plot')
plt.legend()
plt.grid(True)
plt.show()
