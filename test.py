import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具
from colour.models import Lab_to_XYZ, XYZ_to_sRGB, XYZ_to_Oklab

def sRGB_to_linear(srgb):
    threshold = 0.04045
    below_threshold = srgb <= threshold
    above_threshold = srgb > threshold
    linear_rgb = np.zeros_like(srgb)
    linear_rgb[below_threshold] = srgb[below_threshold] / 12.92
    linear_rgb[above_threshold] = ((srgb[above_threshold] + 0.055) / 1.055) ** 2.4
    return linear_rgb

def linear_to_srgb(linear_rgb):
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * (linear_rgb ** (1 / 2.4)) - 0.055
    )
    return srgb

# 定义 Lab 色值
lab_values = np.array([
    [18.3, 13.2, -12.2],
    [39.1, 28.3, 16],
    [44.1, 23.9, 18.6],
    [61.2, -4.8, 42],
    [38.5, -19, 5.6],
    [26.9, 8.9, -22.9],
    [66.7, -1.3, 1.3]
])

# 转换到 sRGB 并转换到线性 RGB
XYZ_values = Lab_to_XYZ(lab_values)
test = XYZ_values

# 定义多个独立的 test[1] 组
test_1_sets = np.array([
   test[1], test[2], test[3], test[4], test[5]
])

# 创建一个空列表来存储插值后的数据点
interpolated_points = []

# 创建一个包含初始点的数组，并插入每个 test[1] 组进行插值
def interpolate_points(data_array, num_points):
    while len(data_array) < num_points:
        new_points = []
        for i in range(len(data_array) - 1):
            midpoint = (data_array[i] + data_array[i + 1]) / 2
            new_points.append(data_array[i])
            new_points.append(midpoint)
        new_points.append(data_array[-1])
        data_array = np.array(new_points)
    return data_array

# 插值并绘制结果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 lab_values 的插值结果
for test_1 in test_1_sets:
    data_array = np.vstack((test[0], test_1, test[6]))
    data_array = interpolate_points(data_array, 30)
    
    # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result = XYZ_to_Oklab(data_array)

    # 提取分量
    L = result[:, 0]
    a = result[:, 1]
    b = result[:, 2]

    # 绘制散点图，点的颜色为自身的归一化值
    ax.scatter(a, b, L, c=result_color, marker='o')
for i in range(len(interpolated_points[0])):
    data_array = np.vstack((interpolated_points[0][i], interpolated_points[1][i], interpolated_points[2][i],
                            interpolated_points[3][i], interpolated_points[4][i], interpolated_points[0][i]))
    data_array = interpolate_points(data_array, 30)

     # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result = XYZ_to_Oklab(data_array)

    # 提取分量
    L = result[:, 0]
    a = result[:, 1]
    b = result[:, 2]

    # 绘制散点图，点的颜色为自身的归一化值
    ax.scatter(a, b, L, c=result_color, marker='o')
    
    
L_zero_line = np.linspace(0, 1, 100)  # 假设 L 的范围是从 0 到 1
ax.plot([0] * len(L_zero_line), [0] * len(L_zero_line), L_zero_line, color='red', linestyle='--', label='a=0, b=0')

# 设置图形参数
ax.set_xlabel('a*')
ax.set_ylabel('b*')
ax.set_zlabel('L*')
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.4, 0.4)
ax.set_zlim(0, 1)
ax.set_title('Interpolated Points in 3D Space')

plt.show()

# 打印所有插值后的数据点
for idx, points in enumerate(interpolated_points):
    print(f"Set {idx + 1}:")
    print(points)
    print("=" * 20)
