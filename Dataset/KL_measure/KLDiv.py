import numpy as np

def KLDiv(P1, P2):
    """
    计算两个离散概率分布的 Kullback-Leibler divergence.
    
    参数:
        P1 (ndarray): 1 x n 数组
        P2 (ndarray): 1 x n 数组
    返回:
        float: KL divergence, 即 sum(P1(i) * (log2(P1(i)) - log2(P2(i))))，
                仅对 P1(i) > 0 且 P2(i) > 0 的分量计算
    """
    P1 = np.asarray(P1).flatten()
    P2 = np.asarray(P2).flatten()
    
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("the number of columns in P1 and P2 should be the same")
    
    if abs(np.sum(P1) - 1) > 1e-5 or abs(np.sum(P2) - 1) > 1e-5:
        raise ValueError("Probabilities don't sum to 1.")
    
    KL = 0.0
    for i in range(len(P1)):
        if P1[i] > 0 and P2[i] > 0:
            KL += P1[i] * (np.log2(P1[i]) - np.log2(P2[i]))
    return KL