"""
CNMF: Coupled Nonnegative Matrix Factorization
Yokoya et al., IEEE TGRS 2012

零训练传统基线方法，直接推理。

算法流程:
    1. VCA 初始化端元矩阵 W
    2. 交替 NMF 分解 HS 和 MS 分支
    3. 通过 PSF (空间) 和 SRF (光谱) 耦合两个分解
    4. 输出 W × H 重建

参考实现:
    - HyperEvalSR (Python port of original MATLAB code)
    - hif-benchmarking (original MATLAB code by Naoto Yokoya)
"""
import numpy as np
import torch
from torch import Tensor

from ...registry import register_model
from ...base_model import BaseFusionModel


# ============================================================
# 工具函数
# ============================================================

def _fspecial_gauss(size: int, sigma: float) -> np.ndarray:
    """生成 2D 高斯核。"""
    x, y = np.mgrid[-size//2+1:size//2+1, -size//2+1:size//2+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / g.sum()


def _gaussian_down_sample(data: np.ndarray, w: int) -> np.ndarray:
    """
    高斯 PSF 下采样。空间耦合用。
    data: (H, W, C) → out: (H//w, W//w, C)
    """
    H, W, C = data.shape
    h_out, w_out = int(np.floor(H / w)), int(np.floor(W / w))
    out = np.zeros((h_out, w_out, C))
    sig = w / 2.35482  # FWHM → sigma

    if w % 2 == 0:
        k1 = _fspecial_gauss(w, sig).reshape(w, w, 1)
        k2 = _fspecial_gauss(w*2, sig).reshape(w*2, w*2, 1)
    else:
        k1 = _fspecial_gauss(w, sig).reshape(w, w, 1)
        k2 = _fspecial_gauss(w*2-1, sig).reshape(w*2-1, w*2-1, 1)

    for x in range(h_out):
        for y in range(w_out):
            # 边缘用单格核，内部用双格核（避免越界）
            if x == 0 or x == h_out-1 or y == 0 or y == w_out-1:
                patch = data[x*w:(x+1)*w, y*w:(y+1)*w, :]
                out[x, y, :] = (patch * k1).sum(axis=0).sum(axis=0)
            else:
                if w % 2 == 0:
                    patch = data[x*w-w//2:(x+1)*w+w//2, y*w-w//2:(y+1)*w+w//2, :]
                else:
                    patch = data[x*w-(w-1)//2:(x+1)*w+(w-1)//2,
                                 y*w-(w-1)//2:(y+1)*w+(w-1)//2, :]
                out[x, y, :] = (patch * k2).sum(axis=0).sum(axis=0)
    return out


def _zoom_bi(data: np.ndarray, w: int) -> np.ndarray:
    """双线性插值上采样 w 倍。"""
    from scipy.ndimage import zoom
    return zoom(data, (w, w), order=1)


def _vca(R: np.ndarray, p: int):
    """
    Vertex Component Analysis (VCA) 端元提取。
    R: (bands, pixels), p: 端元数
    返回: U: (bands, p), indices: list
    """
    N = R.shape[1]
    L = R.shape[0]

    # SNR 估计
    r_m = R.mean(axis=1, keepdims=True)
    R_o = R - r_m
    U, S, V = np.linalg.svd(R_o @ R_o.T / N, full_matrices=False)
    Ud = U[:, :p]
    x_p = Ud.T @ R_o
    P_y = (R**2).sum() / N
    P_x = (x_p**2).sum() / N + (r_m**2).sum()
    SNR = abs(10 * np.log10((P_x - (p/L)*P_y) / (P_y - P_x + 1e-30)))

    SNRth = 15 + 10 * np.log(p) + 8  # MATLAB 原版阈值

    if SNR > SNRth:
        d = p
        Ud, _, _ = np.linalg.svd(R @ R.T / N, full_matrices=False)
        Ud = Ud[:, :d]
        X = Ud.T @ R
        u = X.mean(axis=1, keepdims=True)
        Y = X / (X * u).sum(axis=0, keepdims=True)
    else:
        d = p - 1
        r_m = R.T.mean(axis=0, keepdims=True).T
        R_o = R - r_m
        Ud, _, _ = np.linalg.svd(R_o @ R_o.T / N, full_matrices=False)
        Ud = Ud[:, :d]
        X = Ud.T @ R_o
        c = np.sqrt((X**2).sum(axis=0).max())
        Y = np.vstack((X, np.full((1, N), c)))

    A = np.zeros((p, p))
    A[:, 0] = 1
    I = np.eye(p)
    indices = []
    for i in range(p):
        w = np.random.rand(p, 1)
        f = (I - A @ np.linalg.pinv(A)) @ w
        f = f / np.linalg.norm(f)
        k = np.abs(f.T @ Y).argmax()
        A[:, i] = Y[:, k].flatten()
        indices.append(k)

    if SNR > SNRth:
        return Ud @ X[:, indices], indices
    else:
        return Ud @ X[:, indices] + r_m, indices


def _vd(data: np.ndarray, alpha: float = 5e-2) -> int:
    """
    Virtual Dimensionality — 估计端元数。
    data: (bands, pixels)
    """
    from scipy.special import erfinv
    N = data.shape[1]
    L = data.shape[0]
    R = data @ data.T / N
    K = np.cov(data)
    e_r = np.sort(np.linalg.eigvalsh(R))[::-1]
    e_k = np.sort(np.linalg.eigvalsh(K))[::-1]
    diff = e_r - e_k
    variance = np.sqrt(2 * (e_r**2 + e_k**2) / N)
    # MATLAB: tau = -norminv(alpha, 0, variance)
    tau = -np.sqrt(2) * variance * float(erfinv(2*alpha - 1))
    return int((diff > tau).sum())


def _lsqnonneg(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    非负最小二乘 min ||y - Ax|| s.t. x >= 0
    """
    from scipy.optimize import nnls
    x, _ = nnls(A, y.flatten())
    return x.reshape(-1, 1)


def _nls_su(Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """逐像素非负最小二乘解混。"""
    n = A.shape[1]
    p = Y.shape[1]
    X = np.zeros((n, p))
    for i in range(p):
        x = _lsqnonneg(Y[:, i:i+1], A)
        X[:, i] = x.flatten()
    return X


def _estimate_srf(HS: np.ndarray, MS: np.ndarray) -> np.ndarray:
    """
    从数据估计 SRF 矩阵 + 偏移项。
    NNLS 求解 min ||Ax-b||² s.t. x>=0。
    返回: (bands1, bands2+1)，最后一列为偏移项。
    """
    bands1 = MS.shape[2]
    bands2 = HS.shape[2]
    w = MS.shape[0] // HS.shape[0]

    hs_flat = np.hstack([HS.reshape(-1, bands2),
                         np.ones((HS.size//bands2, 1))])
    ms_down = _gaussian_down_sample(MS, w)
    ms_flat = ms_down.reshape(-1, bands1)

    R = np.zeros((bands1, bands2 + 1))
    for b in range(bands1):
        R[b, :] = _lsqnonneg(ms_flat[:, b:b+1], hs_flat).flatten()
    return R


# ============================================================
# CNMF 核心算法
# ============================================================

def _cnmf_core(hr_msi: np.ndarray, lr_hsi: np.ndarray,
               n_endmembers: int = None,
               inner_iter: int = 200, outer_iter: int = 1,
               th_h: float = 1e-8, th_m: float = 1e-8, th2: float = 1e-2,
               init_mode: int = 0,
               verbose: bool = False) -> np.ndarray:
    """
    CNMF 算法主流程。

    Args:
        hr_msi: (H_hr, W_hr, C_msi)  高分辨率多光谱
        lr_hsi: (H_lr, W_lr, C_hsi)  低分辨率高光谱

    Returns:
        sr_hsi: (H_hr, W_hr, C_hsi)  融合结果
    """
    H_hr, W_hr, C_msi = hr_msi.shape
    H_lr, W_lr, C_hsi = lr_hsi.shape
    w = H_hr // H_lr  # 空间尺度倍数

    # ---- 1. 估计 SRF ----
    if verbose:
        print("  Estimating SRF...")
    R = _estimate_srf(lr_hsi, hr_msi)
    # 减去偏移项
    for b in range(C_msi):
        msi = hr_msi[:, :, b].copy()
        msi -= R[b, -1]
        msi[msi < 0] = 0
        hr_msi[:, :, b] = msi
    R = R[:, :-1]  # 去掉偏移列

    # ---- 2. 参数设置 ----
    sum2one = 2 * (hr_msi.mean() / 0.7455) ** 0.5 / C_msi ** 3

    I1, I2 = inner_iter, outer_iter
    if C_msi == 1:
        I1 = min(I1, 75)

    # ---- 3. 矩阵化 ----
    # HS: (bands, pixels), MS: (bands, pixels)
    Vh = lr_hsi.reshape(-1, C_hsi).T   # (C_hsi, N_lr)
    Vm = hr_msi.reshape(-1, C_msi).T    # (C_msi, N_hr)

    # ---- 4. 估计端元数 ----
    if n_endmembers is not None:
        M = n_endmembers
    else:
        M_est = _vd(Vh, 5e-2)
        M = max(min(30, C_hsi), M_est)
    if verbose:
        print(f"  Endmembers: {M}" + (f" (VD={M_est})" if n_endmembers is None else ""))

    # ---- 5. 初始化 ----
    if verbose:
        print("  Initialization...")

    # VCA 初始化端元
    W_hyper, _ = _vca(Vh, M)

    # sum-to-one 约束
    W_hyper = np.vstack([W_hyper, sum2one * np.ones((1, M))])
    Vh = np.vstack([Vh, sum2one * np.ones((1, Vh.shape[1]))])

    # 初始化丰度矩阵
    if init_mode == 1:
        H_hyper = _nls_su(Vh, W_hyper)
    else:
        H_hyper = np.ones((M, Vh.shape[1])) / M

    # ---- 6. HS 分支 NMF (初始化) ----
    if verbose:
        print("  NMF for HS branch (init)...")
    for i in range(I1):
        if i == 0:
            # 只更新 H_hyper（W_hyper 固定）
            cost0 = 0
            for q in range(I1 * 3):
                H_hyper_old = H_hyper.copy()
                H_n = W_hyper.T @ Vh
                H_d = (W_hyper.T @ W_hyper) @ H_hyper
                H_hyper = H_hyper * H_n / np.maximum(H_d, 1e-30)
                cost = np.linalg.norm(
                    Vh[:C_hsi] - W_hyper[:C_hsi] @ H_hyper, 'fro')**2
                if q > 1 and (cost0 - cost) / max(cost, 1e-30) < th_h:
                    H_hyper = H_hyper_old
                    break
                cost0 = cost
        else:
            # 更新 W_hyper
            W_hy_old = W_hyper.copy()
            W_n = Vh[:C_hsi] @ H_hyper.T
            W_d = (W_hyper[:C_hsi] @ H_hyper) @ H_hyper.T
            W_hyper[:C_hsi] = W_hyper[:C_hsi] * W_n / np.maximum(W_d, 1e-30)
            # 更新 H_hyper
            H_hy_old = H_hyper.copy()
            H_n = W_hyper.T @ Vh
            H_d = (W_hyper.T @ W_hyper) @ H_hyper
            H_hyper = H_hyper * H_n / np.maximum(H_d, 1e-30)
            cost = np.linalg.norm(
                Vh[:C_hsi] - W_hyper[:C_hsi] @ H_hyper, 'fro')**2
            if (cost0 - cost) / max(cost, 1e-30) < th_h:
                W_hyper = W_hy_old
                H_hyper = H_hy_old
                break
            cost0 = cost

    RMSE_h = np.sqrt(cost0 / (Vh.shape[1] * C_hsi))

    # ---- 7. MS 分支初始化 ----
    W_multi = R @ W_hyper[:C_hsi, :]
    W_multi = np.vstack([W_multi, sum2one * np.ones((1, M))])
    Vm = np.vstack([Vm, sum2one * np.ones((1, Vm.shape[1]))])

    # 插值初始丰度
    H_multi = np.ones((M, Vm.shape[1])) / M
    for i in range(M):
        tmp = _zoom_bi(H_hyper[i, :].reshape(H_lr, W_lr), w)
        H_multi[i, :] = tmp.reshape(1, Vm.shape[1])
    H_multi[H_multi < 0] = 0

    # ---- 8. MS 分支 NMF (初始化) ----
    if verbose:
        print("  NMF for MS branch (init)...")
    for i in range(I1):
        if i == 0:
            cost0 = 0
            for q in range(I1):
                H_multi_old = H_multi.copy()
                H_n = W_multi.T @ Vm
                H_d = (W_multi.T @ W_multi) @ H_multi
                H_multi = H_multi * H_n / np.maximum(H_d, 1e-30)
                cost = np.linalg.norm(
                    Vm[:C_msi] - W_multi[:C_msi] @ H_multi, 'fro')**2
                if q > 1 and (cost0 - cost) / max(cost, 1e-30) < th_m:
                    H_multi = H_multi_old
                    break
                cost0 = cost
        else:
            W_multi_old = W_multi.copy()
            if C_msi > 3:
                W_multi_n = Vm[:C_msi] @ H_multi.T
                W_multi_d = (W_multi[:C_msi] @ H_multi) @ H_multi.T
                W_multi[:C_msi] = W_multi[:C_msi] * W_multi_n / np.maximum(W_multi_d, 1e-30)
            H_multi_old = H_multi.copy()
            H_n = W_multi.T @ Vm
            H_d = (W_multi.T @ W_multi) @ H_multi
            H_multi = H_multi * H_n / np.maximum(H_d, 1e-30)
            cost = np.linalg.norm(
                Vm[:C_msi] - W_multi[:C_msi] @ H_multi, 'fro')**2
            if (cost0 - cost) / max(cost, 1e-30) < th_m:
                W_multi = W_multi_old
                H_multi = H_multi_old
                break
            cost0 = cost

    RMSE_m = np.sqrt(cost0 / (Vm.shape[1] * C_msi))

    # ---- 9. 外循环迭代 ----
    cost_h = [RMSE_h]
    cost_m = [RMSE_m]

    for i_out in range(I2):
        if verbose:
            print(f"  Outer iteration {i_out+1}/{I2}...")

        # 空间耦合: H_multi → H_hyper (高斯下采样)
        H_hyper_new = np.zeros((M, Vh.shape[1]))
        for i in range(M):
            # 单通道丰度图 → 加通道维 → 下采样 → 去掉通道维
            hm = H_multi[i, :].reshape(H_hr, W_hr)[:, :, None]
            tmp = _gaussian_down_sample(hm, w)
            H_hyper_new[i, :] = tmp.reshape(1, -1)
        H_hyper = H_hyper_new

        # ---- HS unmixing ----
        cost0 = 0
        for i in range(I1):
            if i == 0:
                # 更新 W_hyper
                for _ in range(I1):
                    W_hy_old = W_hyper.copy()
                    W_n = Vh[:C_hsi] @ H_hyper.T
                    W_d = (W_hyper[:C_hsi] @ H_hyper) @ H_hyper.T
                    W_hyper[:C_hsi] = (W_hyper[:C_hsi] * W_n /
                                       np.maximum(W_d, 1e-30))
                    cost = np.linalg.norm(
                        Vh[:C_hsi] - W_hyper[:C_hsi] @ H_hyper, 'fro')**2
                    if (cost0 - cost) / max(cost, 1e-30) < th_h:
                        W_hyper = W_hy_old
                        break
                    cost0 = cost
            else:
                # 更新 H_hyper 和 W_hyper
                if C_msi > 3:
                    H_hyper = H_hyper * (W_hyper.T @ Vh) / \
                        np.maximum((W_hyper.T @ W_hyper) @ H_hyper, 1e-30)
                H_hy_old = H_hyper.copy()
                W_hy_old = W_hyper.copy()
                W_n = Vh[:C_hsi] @ H_hyper.T
                W_d = (W_hyper[:C_hsi] @ H_hyper) @ H_hyper.T
                W_hyper[:C_hsi] = W_hyper[:C_hsi] * W_n / \
                    np.maximum(W_d, 1e-30)
                cost = np.linalg.norm(
                    Vh[:C_hsi] - W_hyper[:C_hsi] @ H_hyper, 'fro')**2
                if (cost0 - cost) / max(cost, 1e-30) < th_h:
                    H_hyper = H_hy_old
                    W_hyper = W_hy_old
                    break
                cost0 = cost

        cost_h.append(np.sqrt(cost0 / (Vh.shape[1] * C_hsi)))

        # ---- MS unmixing (论文要求交替优化 HS 和 MS) ----
        W_multi[:C_msi, :] = R @ W_hyper[:C_hsi, :]

        cost0 = 0
        for i in range(I1):
            if i == 0:
                for _ in range(I1):
                    H_multi_old = H_multi.copy()
                    H_n = W_multi.T @ Vm
                    H_d = (W_multi.T @ W_multi) @ H_multi
                    H_multi = H_multi * H_n / np.maximum(H_d, 1e-30)
                    cost = np.linalg.norm(
                        Vm[:C_msi] - W_multi[:C_msi] @ H_multi, 'fro')**2
                    if (cost0 - cost) / max(cost, 1e-30) < th_m:
                        H_multi = H_multi_old
                        break
                    cost0 = cost
            else:
                if C_msi > 3:
                    W_multi[:C_msi] = (W_multi[:C_msi] *
                        (Vm[:C_msi] @ H_multi.T)) / \
                        np.maximum((W_multi[:C_msi] @ H_multi) @
                                   H_multi.T, 1e-30)
                H_multi_old = H_multi.copy()
                H_n = W_multi.T @ Vm
                H_d = (W_multi.T @ W_multi) @ H_multi
                H_multi = H_multi * H_n / np.maximum(H_d, 1e-30)
                cost = np.linalg.norm(
                    Vm[:C_msi] - W_multi[:C_msi] @ H_multi, 'fro')**2
                if (cost0 - cost) / max(cost, 1e-30) < th_m:
                    H_multi = H_multi_old
                    break
                cost0 = cost

        cost_m.append(np.sqrt(cost0 / (Vm.shape[1] * C_msi)))

        # 收敛判断
        if i_out < I2 - 1:
            if ((cost_h[-2] - cost_h[-1]) / max(cost_h[-1], 1e-30) > th2
                    and (cost_m[-2] - cost_m[-1]) / max(cost_m[-1], 1e-30) > th2):
                continue
            else:
                if verbose:
                    print("  Converged!")
                break

    # ---- 10. 重建 ----
    sr = (W_hyper[:C_hsi, :] @ H_multi).T.reshape(H_hr, W_hr, C_hsi)
    return sr


# ============================================================
# 框架模型
# ============================================================

@register_model("cnmf")
class CNMF(BaseFusionModel):
    """
    Coupled Nonnegative Matrix Factorization (Yokoya 2012).
    零训练传统方法，直接推理。
    """

    training_required: bool = False
    training_mode: str = "none"

    def __init__(self, config: dict):
        super().__init__(config)
        self.verbose = config.get("verbose", False)

        # CNMF 超参数（全部有默认值，与 MATLAB 原版一致）
        self.n_endmembers = config.get("n_endmembers", None)
        self.inner_iter = config.get("inner_iter", 200)
        self.outer_iter = config.get("outer_iter", 1)
        self.th_h = config.get("th_h", 1e-8)
        self.th_m = config.get("th_m", 1e-8)
        self.th2 = config.get("th2", 1e-2)
        self.init_mode = config.get("init_mode", 0)

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        """
        直接推理，无需训练。

        Args:
            lr_hsi: (B, C_hsi, H_lr, W_lr)
            hr_msi: (B, C_msi, H_hr, W_hr)

        Returns:
            hr_hsi: (B, C_hsi, H_hr, W_hr)
        """
        B = lr_hsi.shape[0]
        results = []

        for b in range(B):
            # Tensor → numpy (H, W, C)
            lr_np = lr_hsi[b].detach().cpu().numpy().transpose(1, 2, 0)
            hr_np = hr_msi[b].detach().cpu().numpy().transpose(1, 2, 0)

            # CNMF 融合
            sr = _cnmf_core(
                hr_np, lr_np,
                n_endmembers=self.n_endmembers,
                inner_iter=self.inner_iter,
                outer_iter=self.outer_iter,
                th_h=self.th_h, th_m=self.th_m, th2=self.th2,
                init_mode=self.init_mode,
                verbose=(self.verbose and b == 0),
            )

            # numpy → Tensor (C, H, W)
            results.append(
                torch.from_numpy(sr.transpose(2, 0, 1)).float()
            )

        return torch.stack(results).to(lr_hsi.device)
