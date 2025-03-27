import numpy as np
from numpy.linalg import pinv, norm

def Bhattacharyya(Y1, Y2):
    """
    计算两个类别的 Bhattacharyya 距离.
    
    参数:
        Y1 (ndarray): 形状为 (num_dim, num_samples) 的数组
        Y2 (ndarray): 形状为 (num_dim, num_samples) 的数组
    返回:
        float: Bhattacharyya 距离
    """
    # 计算均值向量（保持为列向量）
    M1 = np.mean(Y1, axis=1, keepdims=True)
    M2 = np.mean(Y2, axis=1, keepdims=True)
    
    # 计算协方差矩阵（MATLAB 中 cov(Y1') 对应 Python 中 cov(Y1.T)）
    Cov1 = np.cov(Y1.T)
    Cov2 = np.cov(Y2.T)
    
    temp = pinv((Cov1 + Cov2) / 2)
    diff = M2 - M1
    term1 = 1/8 * np.dot(diff.T, np.dot(temp, diff))[0, 0]
    term2 = 1/2 * np.log(norm((Cov1 + Cov2) / 2) / np.sqrt(norm(Cov1) * norm(Cov2)))
    
    y = term1 + term2
    return y