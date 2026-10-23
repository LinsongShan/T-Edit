import torch
import math

def dct_1d(x, dim=-1):
    N = x.size(dim)
    device = x.device
    dtype = x.dtype

    v = torch.cat([x, x.flip(dims=[dim])], dim=dim)
    V = torch.fft.fft(v, dim=dim)

    k = torch.arange(N, device=device, dtype=dtype)
    factor = torch.exp(-1j * math.pi * k / (2 * N))
    return (V.index_select(dim, torch.arange(N, device=device)) * factor).real * math.sqrt(2 / N)


def idct_1d(X, dim=-1):
    N = X.size(dim)
    device = X.device

    X = X.float()

    k = torch.arange(N, device=device, dtype=X.dtype)
    factor = torch.exp(1j * math.pi * k / (2 * N))

    V = torch.zeros(
        list(X.shape[:-1]) + [2 * N],
        dtype=torch.complex64,
        device=device
    )

    V.index_copy_(
        dim,
        torch.arange(N, device=device),
        X * factor
    )

    V.index_copy_(
        dim,
        torch.arange(N, 2 * N, device=device),
        torch.zeros_like(X, dtype=V.dtype)
    )

    v = torch.fft.ifft(V, dim=dim).real
    return v.index_select(dim, torch.arange(N, device=device)) * math.sqrt(2 / N)


def t_svd_low_freq(qkv_tensor, freq_ratio=0.3, rank_ratio=0.5):
    orig_dtype = qkv_tensor.dtype
    device = qkv_tensor.device

    X = qkv_tensor.float()   # SVD / FFT 用 fp32

    H, L, D = X.shape
    k_freq = max(1, int(freq_ratio * D))
    r = max(1, int(rank_ratio * min(H, L)))

    # DCT along D
    X_dct = dct_1d(X, dim=2)

    # keep low freq
    X_dct[..., k_freq:] = 0.0

    # per-frequency low-rank SVD
    for k in range(k_freq):
        U, s, Vh = torch.linalg.svd(X_dct[:, :, k], full_matrices=False)
        X_dct[:, :, k] = (U[:, :r] * s[:r]) @ Vh[:r]

    # IDCT
    X_low = idct_1d(X_dct, dim=2)

    return X_low.to(device=device, dtype=orig_dtype)

def t_svd_low_freq_fft(
    qkv_tensor,
    freq_ratio=0.3,
    rank_ratio=0.5
):
    orig_dtype = qkv_tensor.dtype
    device = qkv_tensor.device

    X = qkv_tensor.float()   # FFT / SVD 用 fp32

    H, L, D = X.shape
    k_freq = max(1, int(freq_ratio * (D // 2 + 1)))
    r = max(1, int(rank_ratio * min(H, L)))

    # FFT along D (real → complex)
    X_fft = torch.fft.rfft(X, dim=2)

    # zero out high frequencies
    X_fft[..., k_freq:] = 0.0

    # per-frequency low-rank SVD
    for k in range(k_freq):
        # complex SVD is supported
        U, s, Vh = torch.linalg.svd(X_fft[:, :, k], full_matrices=False)
        X_fft[:, :, k] = (U[:, :r] * s[:r]) @ Vh[:r]

    # inverse FFT
    X_low = torch.fft.irfft(X_fft, n=D, dim=2)

    return X_low.to(device=device, dtype=orig_dtype)


def layer_importance(tensor, layer_rank=5):
    L, H, S, D = tensor.shape
    reshaped = tensor.reshape(L, -1).to(torch.float32)

    U, s = randomized_svd1(
        reshaped,
        rank=layer_rank,
        oversample=5,
        n_iter=2
    )

    U = U.to(torch.bfloat16)
    s = s.to(torch.bfloat16)

    weights = s / s.sum()

    layer_scores = torch.zeros(L, device=tensor.device)
    for i in range(layer_rank):
        layer_scores += weights[i] * torch.abs(U[:, i])

    total = layer_scores.sum()
    if total > 1e-6:
        importance = layer_scores / total
    else:
        importance = torch.ones(L, device=tensor.device) / L

    return importance.tolist()

def randomized_svd1(X, rank, oversample=5, n_iter=2):
    
    X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    L, N = X.shape
    C = X @ X.T 
    
    # C = U S^2 U.T
    S2, U = torch.linalg.eigh(C)
    
    S2 = torch.flip(S2, dims=[0])
    U = torch.flip(U, dims=[1])
    
    S = torch.sqrt(torch.clamp(S2, min=1e-9))
    
    return U[:, :rank], S[:rank]

def randomized_svd(X, rank, oversample=5, n_iter=2):
    # X: (L, N)
    L, N = X.shape
    device = X.device
    dtype = X.dtype

    k = rank + oversample

    # Step 1: random projection
    Omega = torch.randn(N, k, device=device, dtype=dtype)
    Y = X @ Omega  # (L, k)

    # Step 2: power iterations (improves accuracy)
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)

    # Step 3: orthonormalize
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (L, k)

    # Step 4: small SVD
    B = Q.T @ X  # (k, N)
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

    U = Q @ U_hat  # (L, k)
    return U[:, :rank], S[:rank]    
    
    
