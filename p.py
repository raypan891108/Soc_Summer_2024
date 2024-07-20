import numpy as np

# 球面坐標轉直角坐標
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# 定義兩個向量的球面坐標[ 0.57753498 -0.04969052  0.049446  ] [  1.00000174e+00   2.28547958e-06  -1.13652666e-04]
r1, theta1, phi1 = 0.57753498, -0.04969052, 0.049446 
r2, theta2, phi2 =  1.00000174e+00, 2.28547958e-06, -1.13652666e-04

# 轉換為直角坐標
A = spherical_to_cartesian(r1, theta1, phi1)
B = spherical_to_cartesian(r2, theta2, phi2)

# 計算外積
C = np.cross(A, B)

print("向量 A:", A)
print("向量 B:", B)
print("外積 A × B:", C)
