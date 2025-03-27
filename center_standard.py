import numpy as np

def center_standard(input):
    """
    对输入进行中心化和标准化

    参数:
        input (ndarray): 原始数据，可以为多维数组
    返回:
        ndarray: 标准化后的数据
    """
    temp = input.flatten()
    sigma = np.cov(temp)
    sigma = np.sqrt(sigma)
    in_mean = np.mean(temp)
    output = (input - in_mean) / sigma
    return output