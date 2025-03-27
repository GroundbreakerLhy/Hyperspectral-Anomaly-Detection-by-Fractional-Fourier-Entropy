import numpy as np

# 全局变量，用于存储计算结果缓存
E_saved = None
p_saved = None

def Disfrft(f, a, p=None):
    """
    计算向量 f 的离散分数傅立叶变换
    
    参数:
        f (ndarray): 输入信号（一维数组）
        a (float): 分数傅立叶变换阶次
        p (optional): 近似阶数，默认取 N/2
    返回:
        y (ndarray): 变换后的信号（复数数组）
    """
    global E_saved, p_saved
    
    # 将输入转换为一维数组
    f = np.asarray(f).flatten()
    N = len(f)
    even = (N % 2 == 0)
    
    # MATLAB: shft = rem((0:N-1)+fix(N/2),N)+1;
    # Python 索引从 0 开始，所以不用加 1
    shft = ((np.arange(N) + (N // 2)) % N)
    
    # 设置默认参数
    if p is None:
        p = N / 2
    p = min(max(2, p), N - 1)
    
    # 调用 dFRFT 函数
    E = dFRFT(N, p)
    
    # 构造指数向量: [0:N-2 N-1+even]
    exponent_vector = np.concatenate((np.arange(0, N - 1), np.array([N - 1 + int(even)])))
    
    # 计算相位因子
    phase_factor = np.exp(-1j * np.pi/2 * a * exponent_vector)
    
    # 计算变换
    # MATLAB: y(shft,1) = E*(exp(-j*pi/2*a*([0:N-2 N-1+even])).' .*(E'*f(shft)));
    inner_prod = np.dot(E.conj().T, f[shft])
    y_shft = np.dot(E, (phase_factor * inner_prod))
    
    # 将结果按照 shft 顺序赋值回 y
    y = np.zeros(N, dtype=complex)
    y[shft] = y_shft
    
    return y

def dFRFT(N, p):
    """
    返回傅立叶变换矩阵的 NxN 特征向量
    
    参数:
        N (int): 矩阵大小
        p (float): 近似阶数
    返回:
        ndarray: 特征向量矩阵
    """
    global E_saved, p_saved
    
    # 检查是否已有缓存结果
    if E_saved is None or len(E_saved) != N or p_saved != p:
        E = make_E(N, p)
        E_saved = E
        p_saved = p
    else:
        E = E_saved
    
    return E

def make_E(N, p):
    """
    构造并返回排序后的特征向量
    
    参数:
        N (int): 矩阵大小
        p (float): 近似阶数
    返回:
        ndarray: 排序后的特征向量矩阵
    """
    # 构造矩阵 H，使用近似阶数 p
    d2 = np.array([1, -2, 1])
    d_p = np.array([1])
    s = 0
    st = np.zeros(N)
    
    for k in range(1, int(p/2) + 1):
        # MATLAB: d_p = conv(d2, d_p);
        d_p = np.convolve(d2, d_p)
        
        # MATLAB: st([N-k+1:N,1:k+1]) = d_p;
        indices = np.concatenate((np.arange(N-k, N), np.arange(0, k+1)))
        st[indices] = d_p
        st[0] = 0
        
        # MATLAB: temp = [1,1:k-1;1,1:k-1]; temp = temp(:)'./[1:2*k];
        temp1 = np.concatenate(([1], np.arange(1, k)))
        temp2 = np.concatenate(([1], np.arange(1, k)))
        temp = np.concatenate((temp1, temp2)) / np.arange(1, 2*k + 1)
        
        # MATLAB: s = s + (-1)^(k-1)*prod(temp)*2*st;
        s = s + ((-1) ** (k-1)) * np.prod(temp) * 2 * st
    
    # 构造循环矩阵 + 对角矩阵
    # MATLAB: col = (0:N-1)'; row = (N:-1:1);
    col = np.arange(N).reshape(-1, 1)
    row = np.arange(N-1, -1, -1)
    
    # MATLAB: idx = col(:,ones(N,1)) + row(ones(N,1),:);
    idx = col + row.reshape(1, -1)
    idx = idx % (2*N - 1)  # 对于超出范围的索引进行循环
    
    # MATLAB: st = [s(N:-1:2).';s(:)];
    st = np.concatenate((s[N-1:0:-1], s))
    
    # MATLAB: H = st(idx) + diag(real(fft(s)));
    H = st[idx]
    H_diag = np.real(np.fft.fft(s))
    H += np.diag(H_diag)
    
    # 构造变换矩阵 V
    r = N // 2
    even = (N % 2 == 0)
    
    # MATLAB: V1 = (eye(N-1) + flipud(eye(N-1))) / sqrt(2);
    V1 = (np.eye(N-1) + np.flipud(np.eye(N-1))) / np.sqrt(2)
    
    # MATLAB: V1(N-r:end,N-r:end) = -V1(N-r:end,N-r:end);
    V1[N-r-1:, N-r-1:] = -V1[N-r-1:, N-r-1:]
    
    # MATLAB: if (even), V1(r,r) = 1; end
    if even:
        V1[r-1, r-1] = 1
    
    # MATLAB: V = eye(N); V(2:N,2:N) = V1;
    V = np.eye(N)
    V[1:, 1:] = V1
    
    # 计算特征向量
    # MATLAB: VHV = V*H*V';
    VHV = np.dot(V, np.dot(H, V.T))
    
    # MATLAB: E = zeros(N);
    E = np.zeros((N, N), dtype=complex)
    
    # MATLAB: Ev = VHV(1:r+1,1:r+1); Od = VHV(r+2:N,r+2:N);
    Ev = VHV[:r+1, :r+1]
    Od = VHV[r+1:, r+1:]
    
    # 计算特征值和特征向量
    # MATLAB: [ve,ee] = eig(Ev); [vo,eo] = eig(Od);
    ee, ve = np.linalg.eig(Ev)
    eo, vo = np.linalg.eig(Od)
    
    # MATLAB: E(1:r+1,1:r+1) = fliplr(ve); E(r+2:N,r+2:N) = fliplr(vo);
    E[:r+1, :r+1] = np.fliplr(ve)
    E[r+1:, r+1:] = np.fliplr(vo)
    
    # MATLAB: E = V*E;
    E = np.dot(V, E)
    
    # 重新排列特征向量
    # MATLAB: ind = [1:r+1;r+2:2*r+2]; ind = ind(:);
    ind1 = np.arange(0, r+1)
    ind2 = np.arange(r+1, 2*r+2)
    ind = np.vstack((ind1, ind2)).T.flatten()
    
    # MATLAB: if (even), ind([N,N+2]) = []; else ind(N+1) = []; end
    if even:
        ind = np.delete(ind, [N-1, N+1])
    else:
        ind = np.delete(ind, N)
    
    # MATLAB: E = E(:,ind');
    E = E[:, ind]
    
    return E