import numpy as np

def RX(X):
    """
    计算 RX 检测器得分。

    参数:
        X (ndarray): 形状为 (N, M) 的二维数组，其中 N 是变量数，M 是样本数

    返回:
        ndarray: 每个样本对应的 RX 得分，形状为 (M,)
    """
    N, M = X.shape
    
    # 计算每行（变量）的均值
    X_mean = np.mean(X, axis=1, keepdims=True)
    
    # 对数据进行中心化
    X_centered = X - X_mean
    
    # 计算协方差矩阵（注意与 MATLAB 中实现的略有不同，除以 M 而不是 M-1）
    Sigma = np.dot(X_centered, X_centered.T) / M
    
    # 计算协方差矩阵的逆
    Sigma_inv = np.linalg.inv(Sigma)
    
    # 计算每个样本的 RX 得分：x' * Sigma_inv * x
    D = np.empty(M)
    for m in range(M):
        x = X_centered[:, m]
        D[m] = np.dot(x, np.dot(Sigma_inv, x))
        
    return D