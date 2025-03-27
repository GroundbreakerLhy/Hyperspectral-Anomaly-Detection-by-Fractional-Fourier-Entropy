import numpy as np
from Dataset.utils.dsfrft import dFRFT  # 假设 dFRFT 及其内部函数已经实现

def Disfrft(f, a, p=None):
    """
    Computes discrete fractional Fourier transform
    of order a of vector f.
    p (optional) is the order of approximation, default N/2.
    
    参数:
        f (ndarray): 输入信号（一维数组）
        a (float): 分数傅立叶变换阶次
        p (optional): 近似阶次，默认取 N/2
    返回:
        y (ndarray): 变换后的信号（复数数组）
    """
    # 将输入转换为一维数组
    f = np.asarray(f).flatten()
    N = len(f)
    even = (N % 2 == 0)
    # MATLAB 中: shft = rem((0:N-1)+fix(N/2), N)+1，转换为 Python（0-indexed）
    shft = ((np.arange(N) + (N // 2)) % N)
    
    if p is None:
        p = N / 2
    p = min(max(2, p), N - 1)
    
    # 调用 dFRFT 函数（假设已实现）
    E = dFRFT(N, p)
    
    # 构造指数向量：对应 MATLAB 中 [0:N-2, N-1+even]
    exponent_vector = np.concatenate((np.arange(0, N - 1), np.array([N - 1 + int(even)])))
    
    # 计算相位因子
    phase_factor = np.exp(-1j * np.pi/2 * a * exponent_vector)
    
    # 计算 E' * f(shft)，其中 E' 表示共轭转置
    inner_prod = E.conj().T @ f[shft]
    y_shft = E @ (phase_factor * inner_prod)
    
    # 将结果按照 shft 顺序赋值回 y
    y = np.empty_like(f, dtype=complex)
    y[shft] = y_shft
    
    return y