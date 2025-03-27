import numpy as np
from get_density_vector import get_density_vector  # 假设已实现
from KLDiv import KLDiv  # 假设已实现

def Multiclass_KL_v1(features, C):
    """
    返回矩阵形式的 KL 距离，大小为 num_class x num_class.
    参数:
        features (ndarray): M x N 数组，其中 N 为波段数，M 为样本数
        C (array-like): 各类别的样本数列表或数组
    返回:
        ndarray: 大小为 (num_class, num_class) 的 KL 距离矩阵
    """
    num_class = len(C)
    cum_C = np.zeros(num_class + 1, dtype=int)
    cum_C[1:] = np.cumsum(C)
    
    KL_matrix = np.ones((num_class, num_class))
    for i in range(len(cum_C) - 1):
        for j in range(len(cum_C) - 1):
            if i != j:
                P_temp = features[cum_C[i]: cum_C[i+1], :]
                Q_temp = features[cum_C[j]: cum_C[j+1], :]
                
                P, Q = get_density_vector(P_temp, Q_temp)
                KL_matrix[i, j] = KLDiv(Q, P)
    return KL_matrix