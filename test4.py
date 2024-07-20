import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定義對數模型
def log_model(x, a, b, c, d):
    # 確保對數模型的輸入為正數
    return a * np.log(np.maximum(b * x + c, 1e-10)) + d

# 定義 LrMin、x0 和 Sigma 的已知值
LrMin_values = np.array([0.05, 0.1, 0.15, 0.2])
x0_values = np.array([53.7, 56.8, 58.2, 60.6])
Sigma_values = np.array([43.0, 40.0, 35.0, 34.5])

# 初始參數值
p0_x0 = [50, 10, 0.1, 50]
p0_Sigma = [50, 10, 0.1, 50]

# 嘗試擬合對數模型
try:
    params_x0, _ = curve_fit(log_model, LrMin_values, x0_values, p0=p0_x0, maxfev=3000)
    params_Sigma, _ = curve_fit(log_model, LrMin_values, Sigma_values, p0=p0_Sigma, maxfev=3000)
except RuntimeError as e:
    print(f"RuntimeError: {e}")
    params_x0, params_Sigma = None, None

if params_x0 is not None and params_Sigma is not None:
    # 計算 LrMin 為 0.25, 0.3, 0.35, 0.4 和 0.45 時的 x0 和 Sigma 值
    LrMin_targets = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    x0_targets = log_model(LrMin_targets, *params_x0)
    Sigma_targets = log_model(LrMin_targets, *params_Sigma)

    # 繪製擬合曲線
    LrMin_interp = np.linspace(0.05, 0.5, 200)
    x0_interp = log_model(LrMin_interp, *params_x0)
    Sigma_interp = log_model(LrMin_interp, *params_Sigma)

    plt.figure(figsize=(12, 6))

    # 繪製 x0 擬合圖
    plt.subplot(1, 2, 1)
    plt.plot(LrMin_values, x0_values, 'o', label='Data points')
    plt.plot(LrMin_interp, x0_interp, '-', label='Logarithmic fit')
    plt.plot(LrMin_targets, x0_targets, 'r*', markersize=10, label='Estimates')
    for i, target in enumerate(LrMin_targets):
        plt.text(target, x0_targets[i], f'{x0_targets[i]:.2f}', fontsize=9, ha='right')
    plt.xlabel('$L_{rMin}$')
    plt.ylabel('$x_0$')
    plt.title('Logarithmic Fit of $x_0$')
    plt.legend()

    # 繪製 Sigma 擬合圖
    plt.subplot(1, 2, 2)
    plt.plot(LrMin_values, Sigma_values, 'o', label='Data points')
    plt.plot(LrMin_interp, Sigma_interp, '-', label='Logarithmic fit')
    plt.plot(LrMin_targets, Sigma_targets, 'r*', markersize=10, label='Estimates')
    for i, target in enumerate(LrMin_targets):
        plt.text(target, Sigma_targets[i], f'{Sigma_targets[i]:.2f}', fontsize=9, ha='right')
    plt.xlabel('$L_{rMin}$')
    plt.ylabel('$\\Sigma$')
    plt.title('Logarithmic Fit of $\\Sigma$')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 輸出計算結果
    print(f"LrMin targets: {LrMin_targets}")
    print(f"Estimated x0 values: {x0_targets}")
    print(f"Estimated Sigma values: {Sigma_targets}")
else:
    print("Fitting failed.")
