import numpy as np
from Dataset.utils.fconv import fconv  # 假设 fconv 已经实现

def interp(x):
    """
    sinc 插值

    参数:
        x (ndarray): 输入一维数组
    返回:
        ndarray: 插值后的数组
    """
    N = len(x)
    # 构造中间插值数组 y，长度为 2*N - 1
    y = np.zeros(2 * N - 1)
    y[0:2 * N - 1:2] = x  # 每隔一个位置赋值

    # 构造 sinc 函数的采样点，等价于 MATLAB 中 [-(2*N-3):(2*N-3)]'/2
    t = np.arange(-(2 * N - 3), (2 * N - 3) + 1) / 2.0
    sinc_vals = np.sinc(t)  # np.sinc(x)=sin(pi*x)/(pi*x)
    
    # 进行卷积操作（假设 fconv 实现与 MATLAB 保持一致）
    xint = fconv(y, sinc_vals)  

    # MATLAB 中 xint = xint(2*N-2:end-2*N+3);
    # 转换为 Python（注意 MATLAB 索引从1开始，Python从0开始）
    start_index = 2 * N - 2 - 1  # 转换为0开始索引
    end_index = len(xint) - (2 * N - 2 - 1)
    xint = xint[start_index:end_index]
    
    return xint