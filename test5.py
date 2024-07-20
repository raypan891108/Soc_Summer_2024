import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour.models import Lab_to_XYZ, XYZ_to_sRGB, XYZ_to_Oklab, Oklab_to_XYZ, sRGB_to_XYZ
import math

def cross(v1, v2):
    return np.array([v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]])

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def lenV(value):
    result = (value[0] ** 2) + (value[1] ** 2) + (value[2] ** 2)
    result = math.sqrt(result)
    return result

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

# 白點對齊
N_values = np.array([0.95047/0.34065467,1.0/0.36236193, 1.08883/ 0.38393722])

# 转换到 XYZ 并规范化
XYZ_values = Lab_to_XYZ(lab_values)
normalized_XYZ_values = np.array([value * N_values for value in XYZ_values])

# 转换到 Oklab
OkLab_values = XYZ_to_Oklab(normalized_XYZ_values)

# ACeP white - black
v1 = OkLab_values[0] - OkLab_values[6]
v2 = np.array([-1, 0, 0], dtype=float) * lenV(v1)

# 内积计算
dot_product = np.dot(v1, v2) / (lenV(v1) * lenV(v2))
angle_degrees = math.degrees(np.arccos(dot_product))

# 外积计算
cross_product = cross(v1, v2) / lenV(cross(v1, v2))

# 旋转矩阵计算
As = np.array([[0, -cross_product[2], cross_product[1]], [cross_product[2], 0, -cross_product[0]], [-cross_product[1], cross_product[0], 0]])
R = np.eye(3) + (math.sin(np.arccos(dot_product)) * As) + (1 - math.cos(np.arccos(dot_product))) * (np.dot(As, As))

# 应用旋转矩阵并对齐
NAA_value = np.array([np.dot(R, value - OkLab_values[6]) + OkLab_values[6] for value in OkLab_values])
test = Oklab_to_XYZ(NAA_value)

# 定义多个独立的 test[1] 组
test_1_sets = np.array([test[1], test[2], test[3], test[4], test[5]])

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
    data_array = interpolate_points(data_array, 70)
    
    # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result_color = np.clip(result_color, 0, 1)
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
    data_array = interpolate_points(data_array, 50)

     # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result_color = np.clip(result_color, 0, 1)
    result = XYZ_to_Oklab(data_array)

    # 提取分量
    L = result[:, 0]
    a = result[:, 1]
    b = result[:, 2]

    # 绘制散点图，点的颜色为自身的归一化值
    ax.scatter(a, b, L, c=result_color, marker='o')

# 绘制L*轴线
L_zero_line = np.linspace(0, 1, 100)
ax.plot([0] * len(L_zero_line), [0] * len(L_zero_line), L_zero_line, color='red', linestyle='--', label='a=0, b=0')

# 设置图形参数
ax.set_xlabel('a*')
ax.set_ylabel('b*')
ax.set_zlabel('L*')
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.4, 0.4)
ax.set_zlim(0, 1)
ax.set_title('Interpolated Points in 3D Space')

# 添加 sRGB 8个极点到图中
sRGB_points = np.array([
    [1, 0, 0],  # Red
    [1, 0, 1],  # Magenta
    [0, 0, 1],  # Blue
    [0, 1, 1],  # Cyan
    [0, 1, 0],  # Green
    [1, 1, 0],  # Yellow
    [1, 1, 1],  # White
    [0, 0, 0],  # Black
])
sRGB_points_1 = np.array([
    [1, 0, 0],  # Red
    [1, 0, 1],  # Magenta
    [0, 0, 1],  # Blue
    [0, 1, 1],  # Cyan
    [0, 1, 0],  # Green
    [1, 1, 0],  # Yellow
    [1, 0, 0]   # Red
])
sRGB_labels = ['Red', 'Magenta', 'Blue', 'Cyan', 'Green', 'Yellow', 'White', 'Black']
# sRGB_labels_1 = ['Red', 'Magenta', 'Blue', 'Cyan', 'Green', 'Yellow', 'Red']
sRGB_points_XYZ = sRGB_to_XYZ(sRGB_points)
sRGB_points_XYZ_1 = sRGB_to_XYZ(sRGB_points_1)
# sRGB_points_Oklab = XYZ_to_Oklab(sRGB_points_XYZ)

# for i, point in enumerate(sRGB_points_Oklab):
#     ax.scatter(point[1], point[2], point[0], color=sRGB_points[i], marker='o', s=100, edgecolor='k', label=sRGB_labels[i])


for test_1 in sRGB_points_XYZ:
    data_array = np.vstack((sRGB_points_XYZ[7], test_1))
    data_array = interpolate_points(data_array, 4000)
    
    # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result_color = np.clip(result_color, 0, 1)
    result = XYZ_to_Oklab(data_array)

    # 提取分量
    L = result[:, 0]
    a = result[:, 1]
    b = result[:, 2]

    # 绘制散点图，点的颜色为自身的归一化值
    ax.scatter(a, b, L, c='yellow', marker='o', s=1, linewidths=0.5)


for test_1 in sRGB_points_XYZ:
    data_array = np.vstack((sRGB_points_XYZ[6], test_1))
    data_array = interpolate_points(data_array, 100)
    
    # 将插值后的数据点添加到列表中
    interpolated_points.append(data_array)

    result_color = XYZ_to_sRGB(data_array)
    result_color = np.clip(result_color, 0, 1)
    result = XYZ_to_Oklab(data_array)

    # 提取分量
    L = result[:, 0]
    a = result[:, 1]
    b = result[:, 2]

    # 绘制散点图，点的颜色为自身的归一化值
    ax.scatter(a, b, L, c='yellow', marker='o', s=1, linewidths=0.5)
    
    

data_array = sRGB_points_XYZ_1
data_array = interpolate_points(data_array, 5000)

# 将插值后的数据点添加到列表中
interpolated_points.append(data_array)

result_color = XYZ_to_sRGB(data_array)
result_color = np.clip(result_color, 0, 1)
result = XYZ_to_Oklab(data_array)

# 提取分量
L = result[:, 0]
a = result[:, 1]
b = result[:, 2]

# 绘制散点图，点的颜色为自身的归一化值
ax.scatter(a, b, L, c='yellow', marker='o', s=1, linewidths=0.5)

ax.legend()
plt.show()
