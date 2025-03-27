import numpy as np
from get_density_vector import get_density_vector  # 假设已实现
from KLDiv import KLDiv  # 假设已实现

def Multiclass_KL(features, C):
    """
    返回数据的 KL 距离
    参数:
        features (ndarray): M x N 数组，其中 N 为波段数，M 为样本数
        C (array-like): 每个类别样本数的列表或数组
    返回:
        float: 所有类别间 KL 距离的最小值
    """
    features = features / np.max(features)
    num_class = len(C)
    # 构造累积样本数，cum_C[0] = 0,后续为 cumsum(C)
    cum_C = np.zeros(num_class + 1, dtype=int)
    cum_C[1:] = np.cumsum(C)
    
    KL_temp = []
    for i in range(len(cum_C) - 1):
        for j in range(i + 1, len(cum_C) - 1):
            # MATLAB 下索引从 cum_C(i)+1 到 cum_C(i+1)，Python 下直接使用 cum_C[i]:cum_C[i+1]
            P_temp = features[cum_C[i]: cum_C[i+1], :]
            Q_temp = features[cum_C[j]: cum_C[j+1], :]
            
            P, Q = get_density_vector(P_temp, Q_temp)
            KL_temp.append(KLDiv(Q, P))
            
    KL_dist = min(KL_temp)
    return KL_dist