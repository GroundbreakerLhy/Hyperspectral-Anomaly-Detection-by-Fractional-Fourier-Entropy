import numpy as np
from fconv import fconv  # 假设 fconv 已经实现
from interp import interp  # 假设 interp 已经实现

def frft(f, a):
    """
    快速分数傅立叶变换
    参数:
        f (ndarray): 信号采样（一维数组）
        a (float): 分数傅立叶变换阶次
    返回:
        ndarray: 变换后的信号
    """
    # 将信号转换为一维列向量
    f = np.asarray(f).flatten()
    N = len(f)
    
    # 计算排列索引：MATLAB中 shft = rem((0:N-1)+fix(N/2), N)+1
    shft = ((np.arange(N) + (N // 2)) % N)
    sN = np.sqrt(N)
    a = a % 4

    # 特殊情况处理
    if np.isclose(a, 0):
        return f
    if np.isclose(a, 2):
        return np.flipud(f)
    if np.isclose(a, 1):
        out = np.empty_like(f, dtype=complex)
        out[shft] = np.fft.fft(f[shft]) / sN
        return out
    if np.isclose(a, 3):
        out = np.empty_like(f, dtype=complex)
        out[shft] = np.fft.ifft(f[shft]) * sN
        return out

    # 降低到区间 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # 一般情况： 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    
    # 增加采样率：在信号 f 前后各填充 N-1 个零，同时中间进行 sinc 插值
    f = np.concatenate([np.zeros(N - 1), interp(f), np.zeros(N - 1)])
    
    # chirp 预乘
    r = np.arange(-2 * N + 2, 2 * N - 2 + 1, 2)  # 等价于 MATLAB: [-(2*N-2)+? 实际范围为 -2*N+2 到 2*N-2，步长为2
    chrp = np.exp(-1j * np.pi / N * (tana2 / 4) * (r ** 2))
    f = chrp * f
    
    c = np.pi / N / sina / 4
    # 构造卷积核：索引从 -(4*N-4) 到 4*N-4，步长为1
    r2 = np.arange(-(4 * N - 4), 4 * N - 4 + 1)
    kernel = np.exp(1j * c * (r2 ** 2))
    Faf = fconv(kernel, f)
    
    # MATLAB: Faf = Faf(4*N-3:8*N-7)*sqrt(c/pi);
    # Python 索引转换：起始索引 = (4*N-3) - 1 = 4*N-4, 结束索引为 8*N-7（切片不包含结束位置）
    Faf = Faf[4 * N - 4 : 8 * N - 7] * np.sqrt(c / np.pi)
    
    # chirp 后乘
    Faf = chrp * Faf
    
    # 归一化常数： MATLAB: Faf = exp(-1i*(1-a)*pi/4)*Faf(N:2:end-N+1);
    # Python 中对应的切片：从索引 N-1 到 len(Faf)-(N-1)（不包含终点），步长为2
    Faf = np.exp(-1j * (1 - a) * np.pi / 4) * Faf[N - 1 : len(Faf) - (N - 1) : 2]
    
    return Faf