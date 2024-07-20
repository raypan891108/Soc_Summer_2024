import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具
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

#白點對齊
# D65 is X: 0.95047, Y: 1.00000, Z: 1.08883
N_values = np.array([0.95047/0.34065467,1.0/0.36236193, 1.08883/ 0.38393722])
# print(N_values)
# 转换到 sRGB 并转换到线性 RGB
XYZ_values = Lab_to_XYZ(lab_values)
normalized_XYZ_values = []
for value in XYZ_values:
    normalized_XYZ_values.append(value * N_values)

normalized_XYZ_values = np.array(normalized_XYZ_values)


OkLab_values = XYZ_to_Oklab(normalized_XYZ_values)

#ACeP white - black
v1 = OkLab_values[0] - OkLab_values[6]
v2 = np.array([-1, 0, 0], dtype=float)
v2 = lenV(v1) * v2
print(v1, v2)
#內積
dot_product = np.dot(v1, v2)
dot_product = dot_product / (lenV(v1) * lenV(v2))
dot_product = np.arccos(dot_product)
angle_degrees = math.degrees(dot_product) 

#外積
cross_product = cross(v1, v2)
cross_product = cross_product / lenV(cross_product)

print('angle_degrees:', angle_degrees, 'cross_product:', cross_product)
As = np.array([[0, -1 * cross_product[2], cross_product[1]],
               [cross_product[2], 0, -1 * cross_product[0]],
               [-1 * cross_product[1], cross_product[0], 0]])
print('As:', As, 'cross_product:', cross_product)
identity_matrix = np.eye(3)

#旋轉矩陣
R = identity_matrix + (math.sin(dot_product) * As) + (1 - math.cos(dot_product)) * (np.dot(As, As))
# R = np.linalg.inv(R)
print('R:', R)
NAA_value = []

for value in OkLab_values[0:6]:
    # print('value:', value)1
    v = value - OkLab_values[6]
    newPoint = np.dot(R, v.T).T 
    print('newPoint:', newPoint)
    newPoint = newPoint + OkLab_values[6]  
    
    NAA_value.append(newPoint)
NAA_value.append(OkLab_values[6])
NAA_value = np.array(NAA_value)
print('NAA_value:', NAA_value)
test = Oklab_to_XYZ(NAA_value)


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

# # 打印所有插值后的数据点
# for idx, points in enumerate(interpolated_points):
#     print(f"Set {idx + 1}:")
#     print(points)
#     print("=" * 20)
