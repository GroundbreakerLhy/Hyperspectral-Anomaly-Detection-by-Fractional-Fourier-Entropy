import numpy as np
from Disfrft import Disfrft  # 假设 Disfrft 已经实现

def DFrFT2D(img, a):
    """
    计算图像的二维离散分数傅立叶变换

    参数:
        img (ndarray): 输入图像（二维数组）
        a (float): 分数傅立叶变换阶次
    返回:
        ndarray: 变换后的二维数组
    """
    # 确保输入为浮点型
    img = np.array(img, dtype=float)
    m, n = img.shape

    # 对每一行进行 Disfrft 变换
    P = np.zeros((m, n), dtype=complex)
    for i in range(m):
        t = Disfrft(img[i, :], a)
        P[i, :] = t

    # 对每一列进行 Disfrft 变换
    Q = np.zeros((m, n), dtype=complex)
    for j in range(n):
        t = Disfrft(P[:, j], a)
        Q[:, j] = t

    return Q