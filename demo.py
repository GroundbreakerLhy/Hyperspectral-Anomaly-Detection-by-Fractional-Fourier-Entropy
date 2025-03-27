import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 从本地导入未实现的函数
from RX import RX
from FRFEorder import FRFEorder
from Dataset.utils.Disfrft import Disfrft
from center_standard import center_standard

import pywt

def dwt(signal, wavelet='db1', mode='sym'):
    """
    使用 PyWavelets 实现离散小波变换

    参数:
        signal (ndarray): 一维信号数组
        wavelet (str): 小波名称，默认 'db1'
        mode (str): 模式，默认 'sym'
    
    返回:
        (ndarray, ndarray): 近似系数 ca 和细节系数 cd
    """
    ca, cd = pywt.dwt(signal, wavelet, mode=mode)
    return np.array(ca), np.array(cd)


# 加载数据
# mat_data = sio.loadmat('Data/abu-airport-4.mat')
# DataTest = mat_data['data']
# DataTest = DataTest / np.max(DataTest)
# mask = mat_data['map'].astype(float)

mat_data = sio.loadmat("Data/grass_test_2.mat")
DataTest = mat_data['Grass']
DataTest = DataTest / np.max(DataTest)
mask = mat_data['gt'].astype(float)

rows, cols, bands = DataTest.shape
M = rows * cols

# 将数据reshape为二维矩阵: 每一行为一个像素，每一列为一个波段
X = DataTest.reshape((M, bands))

# Global RX: 注意 RX 接口要求输入形状为 (variables, samples)
r0 = RX(X.T)

# Derivative RX
# MATLAB: X2 = abs(X(:,5:end) - X(:,1:end-4));
X2 = np.abs(X[:, 4:] - X[:, :bands-4])
r1 = RX(X2.T)

# DWT-RX
# MATLAB: 对每个样本做离散小波变换，仅取近似系数 ca
XW_list = []
for i in range(M):
    ca, cd = dwt(X[i, :], wavelet='db1', mode='sym')
    XW_list.append(ca)
XW = np.array(XW_list)
r3 = RX(XW.T)

# FrFT + RX
# MATLAB: [FrFE,order] = FRFEorder(DataTest);
FrFE, order = FRFEorder(DataTest)
im1 = np.zeros((rows, cols, bands), dtype=complex)
for i in range(rows):
    for j in range(cols):
        pixel_spectrum = DataTest[i, j, :]
        im1[i, j, :] = Disfrft(pixel_spectrum, order)
im1 = center_standard(np.abs(im1))
XT = im1.reshape((M, bands))
r2 = RX(XT.T)

# DFT + RX
im1 = np.zeros((rows, cols, bands), dtype=complex)
for i in range(rows):
    for j in range(cols):
        pixel_spectrum = DataTest[i, j, :]
        im1[i, j, :] = np.fft.fft(pixel_spectrum)
im1 = center_standard(np.abs(im1))
XT = im1.reshape((M, bands))
r5 = RX(XT.T)

# 计算 ROC
print('Running ROC...')
mask = mask.reshape((1, M))
anomaly_map = (mask == 1)
normal_map = (mask == 0)

def compute_ROC(r, normal_map, anomaly_map):
    r_min = np.min(r)
    r_max = np.max(r)
    taus = np.linspace(r_min, r_max, 5000)
    PF = []
    PD = []
    for tau in taus:
        anomaly_map_rx = (r > tau)
        PF_val = np.sum(np.logical_and(anomaly_map_rx, normal_map)) / np.sum(normal_map)
        PD_val = np.sum(np.logical_and(anomaly_map_rx, anomaly_map)) / np.sum(anomaly_map)
        PF.append(PF_val)
        PD.append(PD_val)
    return np.array(PF), np.array(PD)

PF0, PD0 = compute_ROC(r0, normal_map, anomaly_map)
PF1, PD1 = compute_ROC(r1, normal_map, anomaly_map)
PF3, PD3 = compute_ROC(r3, normal_map, anomaly_map)
PF2, PD2 = compute_ROC(r2, normal_map, anomaly_map)
PF5, PD5 = compute_ROC(r5, normal_map, anomaly_map)

# 计算面积，采用梯形积分法
area0 = np.sum((PF0[:-1] - PF0[1:]) * (PD0[1:] + PD0[:-1]) / 2)
area1 = np.sum((PF1[:-1] - PF1[1:]) * (PD1[1:] + PD1[:-1]) / 2)
area3 = np.sum((PF3[:-1] - PF3[1:]) * (PD3[1:] + PD3[:-1]) / 2)
area2 = np.sum((PF2[:-1] - PF2[1:]) * (PD2[1:] + PD2[:-1]) / 2)
area5 = np.sum((PF5[:-1] - PF5[1:]) * (PD5[1:] + PD5[:-1]) / 2)

# 绘制 ROC 曲线
plt.figure()
plt.plot(PF0, PD0, 'r-.', linewidth=2, label='Global-RX')
plt.plot(PF5, PD5, 'm-', linewidth=2, label='DFT-RX')
plt.plot(PF3, PD3, 'c-', linewidth=2, label='DWT-RX')
plt.plot(PF1, PD1, 'b--', linewidth=2, label='Deriv-RX')
plt.plot(PF2, PD2, 'k-', linewidth=2, label='FrFT-RX')
plt.xlabel('Probability of false alarm')
plt.ylabel('Probability of detection')
plt.legend()
plt.grid(True)
plt.axis([0, 0.3, 0.8, 1])
plt.show()