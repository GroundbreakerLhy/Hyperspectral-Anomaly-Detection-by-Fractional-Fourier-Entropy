import numpy as np
from Dataset.utils.Disfrft import Disfrft
from center_standard import center_standard

def entropy(image):
    """
    计算图像的熵值
    
    参数:
        image (ndarray): 二维图像数组
    返回:
        float: 图像的熵值
    """
    # 将图像转换为浮点型并展平为一维数组
    image_array = np.asarray(image, dtype=float).flatten()
    
    # 归一化到 [0,1] 范围
    if np.max(image_array) != np.min(image_array):
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # 计算直方图
    hist, _ = np.histogram(image_array, bins=256, range=(0, 1), density=True)
    
    # 移除零概率
    hist = hist[hist > 0]
    
    # 计算熵: -sum(p*log2(p))
    return -np.sum(hist * np.log2(hist))

def FRFEorder(DataTest):
    """
    根据 DataTest 计算 FrFE 及其对应的阶数 order

    参数:
        DataTest (ndarray): 形状为 (rows, cols, bands) 的数据
    返回:
        FrFE: 最大的 FrFE 得分
        order: 对应的阶数（除以 10）
    """
    rows, cols, bands = DataTest.shape
    E = np.zeros((11, bands))
    index = 0
    for p in np.arange(0, 1.0 + 0.1, 0.1):
        im1 = np.zeros((rows, cols, bands), dtype=complex)
        for i in range(rows):
            for j in range(cols):
                # DataTest[i,j,:] 已为一维数组
                im1[i, j, :] = Disfrft(DataTest[i, j, :], p)
        im1 = center_standard(np.abs(im1))
        for ii in range(bands):
            E[index, ii] = entropy(im1[:, :, ii])
        index += 1
    # 找出每一行最大值，再在所有行中找全局最大值和对应的索引
    row_max = E.max(axis=1)
    order_index = np.argmax(row_max)
    FrFE = row_max[order_index]
    order = order_index / 10.0
    return FrFE, order