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

# 計算累積和
cumulative_sum = np.cumsum(s_values)

# 繪製圖形
plt.figure(figsize=(10, 6))
plt.plot(n_values, cumulative_sum, color='orange', linestyle='--', label='Cumulative Sum of S_function(n)')
plt.xlabel('n')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of S_function(n)')
plt.legend()
plt.grid(False)  # 關閉網格線
plt.show()
