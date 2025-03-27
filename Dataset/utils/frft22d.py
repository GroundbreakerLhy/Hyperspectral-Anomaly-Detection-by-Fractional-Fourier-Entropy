import numpy as np
from fconv import fconv  # 假设 fconv 已经实现

def frft22d(matrix, angles):
    """
    计算给定矩阵的 2-D FRFT，使用给定角度
    参数:
        matrix (ndarray): 待变换矩阵
        angles (sequence): [角度_x, 角度_y]
    返回:
        out (ndarray): 2-D FRFT 后的矩阵
    """
    # 转换为浮点型并转置
    temp = np.array(matrix, dtype=float).T

    # ----------------- x 方向变换 -----------------
    N = matrix.shape[0]
    a = angles[0] % 4
    shft = ((np.arange(N) + (N // 2)) % N)  # python索引从0开始
    sN = np.sqrt(N)

    if np.isclose(a, 0):
        pass  # do nothing
    elif np.isclose(a, 2):
        temp = np.flipud(temp)
    elif np.isclose(a, 1):
        temp[shft, :] = np.fft.fft(temp[shft, :], axis=0) / sN
    elif np.isclose(a, 3):
        temp[shft, :] = np.fft.ifft(temp[shft, :], axis=0) * sN
    else:
        # reduce to interval 0.5 < a < 1.5
        if a > 2.0:
            a = a - 2
            temp = np.flipud(temp)
        elif a > 1.5:
            a = a - 1
            temp[shft, :] = np.fft.fft(temp[shft, :], axis=0) / sN
        elif a < 0.5:
            a = a + 1
            temp[shft, :] = np.fft.ifft(temp[shft, :], axis=0) * sN

        alpha = a * np.pi / 2
        s = np.pi / (N + 1) / np.sin(alpha) / 4
        t = np.pi / (N + 1) * np.tan(alpha / 2) / 4
        Cs = np.sqrt(s / np.pi) * np.exp(-1j * (1 - a) * np.pi / 4)

        # 构造 sinc 插值窗口
        snc_arg = np.arange(-(2 * N - 3), (2 * N - 3) + 1, 2) / 2.0
        snc = np.sinc(snc_arg)
        # 构造 chirp 向量
        chrp = np.exp(-1j * t * (np.arange(-N + 1, N) ** 2))
        chrp2 = np.exp(1j * s * (np.arange(-(2 * N - 1), (2 * N - 1) + 1) ** 2))

        # 对每一列做变换
        for ix in range(N):
            f0 = temp[:, ix]
            f1 = fconv(f0, snc, 1)
            # MATLAB: f1 = f1(N:2*N-2);
            f1 = f1[N - 1:2 * N - 1]

            l0 = chrp[0::2]  # 奇数索引（MATLAB 1:2:end）
            l1 = chrp[1::2]  # 偶数索引（MATLAB 2:2:end）
            f0 = f0 * l0
            f1 = f1 * l1

            e1 = chrp2[0::2]
            e0 = chrp2[1::2]
            f0 = fconv(f0, e0, 0)
            f1 = fconv(f1, e1, 0)
            h0 = np.fft.ifft(f0 + f1)

            # MATLAB: temp(:,ix) = Cs*l0.*h0(N:2*N-1);
            temp[:, ix] = Cs * l0 * h0[N - 1:2 * N]
    
    # 转置回去
    temp = temp.T

    # ----------------- y 方向变换 -----------------
    N = matrix.shape[1]
    a = angles[1] % 4
    shft = ((np.arange(N) + (N // 2)) % N)
    sN = np.sqrt(N)

    if np.isclose(a, 0):
        pass
    elif np.isclose(a, 2):
        temp = np.flipud(temp)
    elif np.isclose(a, 1):
        temp[shft, :] = np.fft.fft(temp[shft, :], axis=0) / sN
    elif np.isclose(a, 3):
        temp[shft, :] = np.fft.ifft(temp[shft, :], axis=0) * sN
    else:
        if a > 2.0:
            a = a - 2
            temp = np.flipud(temp)
        elif a > 1.5:
            a = a - 1
            temp[shft, :] = np.fft.fft(temp[shft, :], axis=0) / sN
        elif a < 0.5:
            a = a + 1
            temp[shft, :] = np.fft.ifft(temp[shft, :], axis=0) * sN

        alpha = a * np.pi / 2
        s = np.pi / (N + 1) / np.sin(alpha) / 4
        t = np.pi / (N + 1) * np.tan(alpha / 2) / 4
        Cs = np.sqrt(s / np.pi) * np.exp(-1j * (1 - a) * np.pi / 4)

        snc_arg = np.arange(-(2 * N - 3), (2 * N - 3) + 1, 2) / 2.0
        snc = np.sinc(snc_arg)
        chrp = np.exp(-1j * t * (np.arange(-N + 1, N) ** 2))
        chrp2 = np.exp(1j * s * (np.arange(-(2 * N - 1), (2 * N - 1) + 1) ** 2))

        for ix in range(N):
            f0 = temp[:, ix]
            f1 = fconv(f0, snc, 1)
            f1 = f1[N - 1:2 * N - 1]

            l0 = chrp[0::2]
            l1 = chrp[1::2]
            f0 = f0 * l0
            f1 = f1 * l1

            e1 = chrp2[0::2]
            e0 = chrp2[1::2]
            f0 = fconv(f0, e0, 0)
            f1 = fconv(f1, e1, 0)
            h0 = np.fft.ifft(f0 + f1)

            temp[:, ix] = Cs * l0 * h0[N - 1:2 * N]
    
    out = temp
    return out