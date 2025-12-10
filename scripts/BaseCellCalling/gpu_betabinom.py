#!/usr/bin/env python3
"""GPU-accelerated beta-binomial calculations for SComatic (CuPy required)."""

import numpy as np

HAS_CUPY = False
cp = None
cupy_betaln = None
cupy_gammaln = None

try:
    import cupy as cp
    from cupyx.scipy.special import betaln as cupy_betaln, gammaln as cupy_gammaln
    cp.cuda.Device(0).compute_capability
    test_arr = cp.array([1.0])
    _ = cp.floor(test_arr)
    HAS_CUPY = True
except Exception as e:
    CUPY_ERROR = str(e)
    HAS_CUPY = False
    cp = None


def get_backend() -> str:
    if not HAS_CUPY:
        raise RuntimeError(
            f"CuPy (GPU) required but not available: {CUPY_ERROR}\n"
            "Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)"
        )
    return "cupy"


def _log_betabinom_pmf_cupy(k, n, alpha: float, beta: float):
    log_comb = cupy_gammaln(n + 1) - cupy_gammaln(k + 1) - cupy_gammaln(n - k + 1)
    log_beta_num = cupy_betaln(k + alpha, n - k + beta)
    log_beta_denom = cupy_betaln(alpha, beta)
    return log_comb + log_beta_num - log_beta_denom


def betabinom_sf_batch(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if not HAS_CUPY:
        raise RuntimeError(
            f"CuPy (GPU) required but not available: {CUPY_ERROR}\n"
            "Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)"
        )

    k = cp.floor(cp.asarray(k, dtype=cp.float64))
    n = cp.asarray(n, dtype=cp.float64)

    result = cp.ones_like(k)
    valid = (k >= 0) & (n > 0)

    if not cp.any(valid):
        return cp.asnumpy(result)

    k_valid = k[valid]
    n_valid = n[valid]
    max_k = int(cp.max(k_valid).get()) + 1

    i_range = cp.arange(max_k, dtype=cp.float64).reshape(-1, 1)
    k_expanded = k_valid.reshape(1, -1)
    n_expanded = n_valid.reshape(1, -1)
    mask = i_range <= k_expanded

    log_pmf = cp.where(mask, _log_betabinom_pmf_cupy(i_range, n_expanded, alpha, beta), -cp.inf)
    log_cdf = cp.logaddexp.reduce(log_pmf, axis=0)
    sf = 1.0 - cp.exp(log_cdf)

    result[valid] = cp.clip(sf, 0, 1)
    return cp.asnumpy(result)


class GPUBetaBinomCalculator:
    def __init__(
        self,
        alpha1: float = 0.260288007167716,
        beta1: float = 173.94711910763732,
        alpha2: float = 0.08354121346569514,
        beta2: float = 103.47683488327257,
    ):
        if not HAS_CUPY:
            raise RuntimeError(
                f"CuPy (GPU) required but not available: {CUPY_ERROR}\n"
                "Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)"
            )
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    def calculate_pvalues(
        self,
        alt_base_counts: np.ndarray,
        depths: np.ndarray,
        alt_cell_counts: np.ndarray,
        total_cells: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bc_pvalues = betabinom_sf_batch(alt_base_counts, depths, self.alpha1, self.beta1)
        cc_pvalues = betabinom_sf_batch(alt_cell_counts, total_cells, self.alpha2, self.beta2)
        return bc_pvalues, cc_pvalues


if __name__ == "__main__":
    print(f"CuPy available: {HAS_CUPY}")
    if HAS_CUPY:
        print("Backend: cupy")
    else:
        print(f"Error: {CUPY_ERROR}")
