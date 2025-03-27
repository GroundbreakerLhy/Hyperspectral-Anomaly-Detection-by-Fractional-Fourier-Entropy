import numpy as np

def get_density_vector(P_temp, Q_temp):
    """
    输入:
        P_temp, Q_temp: 输入数据，可以为任意形状，最终转换为一维数组。
    返回:
        P, Q: 两个概率密度向量，用于 KL 距离计算
    """
    # 将输入转换为一维数组（行向量）
    P_temp = np.asarray(P_temp).flatten()
    Q_temp = np.asarray(Q_temp).flatten()
    
    # 根据 MATLAB 代码:
    # N = size([P_temp; Q_temp], 1)/2
    # 因为 [P_temp; Q_temp] 在 MATLAB 中为 2 x L 矩阵, 故 N = L.
    N = len(P_temp)
    
    # 计算直方图的 bin 数
    bin_count = int(np.floor(1 + np.log2(N) + 0.5))
    
    # 计算数据的最小值和最大值
    z_min = np.min(np.concatenate([P_temp, Q_temp]))
    z_max = np.max(np.concatenate([P_temp, Q_temp]))
    
    # 构造 bin 边界，等价于 MATLAB 中 z_min: (z_max-z_min)/(bin_count-1) : z_max
    bins = np.linspace(z_min, z_max, num=bin_count)
    
    # 计算直方图计数（注意 np.histogram 默认不包含右边界，近似于 MATLAB histc）
    P_hist, _ = np.histogram(P_temp, bins=bins)
    Q_hist, _ = np.histogram(Q_temp, bins=bins)
    
    # 转换为概率密度并加上 eps 防止数值为 0
    eps = np.finfo(float).eps
    P = P_hist / np.sum(P_hist) + eps
    Q = Q_hist / np.sum(Q_hist) + eps
    
    return P, Q