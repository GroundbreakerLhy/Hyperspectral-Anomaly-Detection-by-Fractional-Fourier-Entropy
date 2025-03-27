import numpy as np
from Bhattacharyya import Bhattacharyya  # 假设 Bhattacharyya 已经实现

def Multiclass_BD_v1(features, C):
    """
    返回矩阵形式的 Bhattacharyya 距离，大小为 num_class x num_class.
    
    参数:
        features (ndarray): M x N 数组，其中 N 为波段数，M 为样本数
        C (array-like): 各类别的样本数列表或数组
    返回:
        ndarray: 大小为 (num_class, num_class) 的 Bhattacharyya 距离矩阵
    """
    # 归一化 features
    features = features / np.max(features)
    
    num_class = len(C)
    cum_C = np.zeros(num_class + 1, dtype=int)
    cum_C[1:] = np.cumsum(C)
    
    BD_matrix = np.ones((num_class, num_class))
    for i in range(len(cum_C) - 1):
        for j in range(len(cum_C) - 1):
            if i != j:
                # Python 索引直接使用 cum_C[i]:cum_C[i+1]
                P_temp = features[cum_C[i]: cum_C[i+1], :]
                Q_temp = features[cum_C[j]: cum_C[j+1], :]
                # 注意：MATLAB 中使用转置，此处转换为 f.T
                BD_matrix[i, j] = Bhattacharyya(P_temp.T, Q_temp.T)
    return BD_matrix