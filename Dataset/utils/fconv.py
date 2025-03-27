import numpy as np

def nextpow2(n):
    """
    计算大于或等于 n 的最小的 2 的幂次指数.
    """
    return int(np.ceil(np.log2(n)))

def fconv(x, y):
    """
    使用 FFT 实现卷积
    参数:
        x (ndarray): 输入向量
        y (ndarray): 输入向量
    返回:
        ndarray: 卷积结果（仅前 N 个样本）
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # 计算 N = len(x) + len(y) - 1，对应 MATLAB 中 [x(:); y(:)] 的长度减 1
    N = len(x) + len(y) - 1
    P = 2 ** nextpow2(N)
    z = np.fft.ifft(np.fft.fft(x, n=P) * np.fft.fft(y, n=P))
    z = z[:N]
    return z