#!/usr/bin/env python3
"""
GPU-Accelerated Beta-Binomial Statistical Testing for SComatic

This module provides massively parallel beta-binomial p-value calculations
using CuPy (NVIDIA GPU) or JAX (CPU/GPU/TPU) as backends.

The beta-binomial survival function P(X >= k) is computed using the
regularized incomplete beta function:
    sf(k, n, α, β) = I_{p}(k, n-k+1) where p = (k+α)/(n+α+β) approximately
    
For exact calculation: sf(k, n, α, β) = 1 - Σ_{i=0}^{k} P(X=i)
We use the beta function relationship for efficiency.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

BACKEND = None
xp = np

try:
    import cupy as cp
    from cupyx.scipy.special import betaln as cupy_betaln
    from cupyx.scipy.special import gammaln as cupy_gammaln
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import betaln as jax_betaln
    from jax.scipy.special import gammaln as jax_gammaln
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from scipy.special import betaln as scipy_betaln
from scipy.special import gammaln as scipy_gammaln
from scipy.stats import betabinom as scipy_betabinom


def get_backend(prefer: str = "cupy") -> str:
    """Select the best available backend."""
    global BACKEND, xp
    
    if prefer == "cupy" and HAS_CUPY:
        BACKEND = "cupy"
        xp = cp
    elif prefer == "jax" and HAS_JAX:
        BACKEND = "jax"
        xp = jnp
    elif HAS_CUPY:
        BACKEND = "cupy"
        xp = cp
    elif HAS_JAX:
        BACKEND = "jax"
        xp = jnp
    else:
        BACKEND = "numpy"
        xp = np
    
    return BACKEND


def _log_betabinom_pmf_numpy(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Compute log of beta-binomial PMF using numpy/scipy.
    log P(X=k|n,α,β) = log C(n,k) + log B(k+α, n-k+β) - log B(α, β)
    """
    log_comb = scipy_gammaln(n + 1) - scipy_gammaln(k + 1) - scipy_gammaln(n - k + 1)
    log_beta_num = scipy_betaln(k + alpha, n - k + beta)
    log_beta_denom = scipy_betaln(alpha, beta)
    return log_comb + log_beta_num - log_beta_denom


def _log_betabinom_pmf_cupy(k, n, alpha: float, beta: float):
    """Compute log of beta-binomial PMF using CuPy."""
    log_comb = cupy_gammaln(n + 1) - cupy_gammaln(k + 1) - cupy_gammaln(n - k + 1)
    log_beta_num = cupy_betaln(k + alpha, n - k + beta)
    log_beta_denom = cupy_betaln(alpha, beta)
    return log_comb + log_beta_num - log_beta_denom


def _log_betabinom_pmf_jax(k, n, alpha: float, beta: float):
    """Compute log of beta-binomial PMF using JAX."""
    log_comb = jax_gammaln(n + 1) - jax_gammaln(k + 1) - jax_gammaln(n - k + 1)
    log_beta_num = jax_betaln(k + alpha, n - k + beta)
    log_beta_denom = jax_betaln(alpha, beta)
    return log_comb + log_beta_num - log_beta_denom


def betabinom_sf_batch_numpy(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Batch compute beta-binomial survival function P(X >= k) using numpy.
    Uses log-sum-exp trick for numerical stability.
    
    For sf(k, n, α, β) = P(X >= k) = 1 - P(X < k) = 1 - Σ_{i=0}^{k-1} P(X=i)
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    
    result = np.ones_like(k)
    valid = (k > 0) & (n > 0) & (k <= n)
    
    if not np.any(valid):
        return result
    
    k_valid = k[valid]
    n_valid = n[valid]
    
    max_k = int(np.max(k_valid))
    n_points = len(k_valid)
    
    i_range = np.arange(max_k).reshape(-1, 1)
    k_expanded = k_valid.reshape(1, -1)
    n_expanded = n_valid.reshape(1, -1)
    
    mask = i_range < k_expanded
    
    log_pmf = np.where(
        mask,
        _log_betabinom_pmf_numpy(i_range, n_expanded, alpha, beta),
        -np.inf
    )
    
    log_cdf = np.logaddexp.reduce(log_pmf, axis=0)
    cdf = np.exp(log_cdf)
    sf = 1.0 - cdf
    
    result[valid] = np.clip(sf, 0, 1)
    return result


def betabinom_sf_batch_cupy(k, n, alpha: float, beta: float):
    """
    Batch compute beta-binomial survival function using CuPy (GPU).
    """
    k = cp.asarray(k, dtype=cp.float64)
    n = cp.asarray(n, dtype=cp.float64)
    
    result = cp.ones_like(k)
    valid = (k > 0) & (n > 0) & (k <= n)
    
    if not cp.any(valid):
        return result
    
    k_valid = k[valid]
    n_valid = n[valid]
    
    max_k = int(cp.max(k_valid).get())
    n_points = len(k_valid)
    
    i_range = cp.arange(max_k, dtype=cp.float64).reshape(-1, 1)
    k_expanded = k_valid.reshape(1, -1)
    n_expanded = n_valid.reshape(1, -1)
    
    mask = i_range < k_expanded
    
    log_pmf = cp.where(
        mask,
        _log_betabinom_pmf_cupy(i_range, n_expanded, alpha, beta),
        -cp.inf
    )
    
    log_cdf = cp.logaddexp.reduce(log_pmf, axis=0)
    cdf = cp.exp(log_cdf)
    sf = 1.0 - cdf
    
    result[valid] = cp.clip(sf, 0, 1)
    return result


def betabinom_sf_batch_jax(k, n, alpha: float, beta: float):
    """
    Batch compute beta-binomial survival function using JAX.
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    
    result = jnp.ones_like(k)
    valid = (k > 0) & (n > 0) & (k <= n)
    
    k_valid = k[valid]
    n_valid = n[valid]
    
    max_k = int(jnp.max(k_valid))
    
    i_range = jnp.arange(max_k, dtype=jnp.float64).reshape(-1, 1)
    k_expanded = k_valid.reshape(1, -1)
    n_expanded = n_valid.reshape(1, -1)
    
    mask = i_range < k_expanded
    
    log_pmf = jnp.where(
        mask,
        _log_betabinom_pmf_jax(i_range, n_expanded, alpha, beta),
        -jnp.inf
    )
    
    log_cdf = jax.scipy.special.logsumexp(log_pmf, axis=0)
    cdf = jnp.exp(log_cdf)
    sf = 1.0 - cdf
    
    result = result.at[valid].set(jnp.clip(sf, 0, 1))
    return result


def betabinom_sf_batch(
    k: np.ndarray, 
    n: np.ndarray, 
    alpha: float, 
    beta: float,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    Batch compute beta-binomial survival function P(X >= k).
    
    Args:
        k: Array of observed counts (alternative allele counts)
        n: Array of total counts (depth or cell counts)
        alpha: Alpha parameter of beta-binomial distribution
        beta: Beta parameter of beta-binomial distribution
        backend: Force specific backend ("cupy", "jax", "numpy")
    
    Returns:
        Array of p-values (survival function values)
    """
    if backend is None:
        backend = BACKEND or get_backend()
    
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    
    if backend == "cupy" and HAS_CUPY:
        result = betabinom_sf_batch_cupy(k, n, alpha, beta)
        return cp.asnumpy(result)
    elif backend == "jax" and HAS_JAX:
        result = betabinom_sf_batch_jax(k, n, alpha, beta)
        return np.asarray(result)
    else:
        return betabinom_sf_batch_numpy(k, n, alpha, beta)


class GPUBetaBinomCalculator:
    """
    High-level interface for GPU-accelerated beta-binomial calculations.
    
    Designed for SComatic variant calling where we need to test:
    1. Base counts (reads) against depth
    2. Cell counts against total cells
    
    Example usage:
        calc = GPUBetaBinomCalculator(alpha1=0.26, beta1=173.9, alpha2=0.08, beta2=103.5)
        
        # Prepare data matrices
        alt_counts = np.array([...])  # Alternative allele counts
        depths = np.array([...])       # Total depths
        cell_counts = np.array([...])  # Cells with alt allele  
        total_cells = np.array([...])  # Total cells
        
        # Batch calculate all p-values at once
        bc_pvalues, cc_pvalues = calc.calculate_pvalues(
            alt_counts, depths, cell_counts, total_cells
        )
    """
    
    def __init__(
        self,
        alpha1: float = 0.260288007167716,
        beta1: float = 173.94711910763732,
        alpha2: float = 0.08354121346569514,
        beta2: float = 103.47683488327257,
        backend: Optional[str] = None,
    ):
        """
        Initialize calculator with beta-binomial parameters.
        
        Args:
            alpha1, beta1: Parameters for base count (read) distribution
            alpha2, beta2: Parameters for cell count distribution
            backend: Force specific backend
        """
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.backend = backend or get_backend()
        
        print(f"GPUBetaBinomCalculator initialized with backend: {self.backend}")
        if self.backend == "cupy":
            print(f"  GPU: {cp.cuda.Device().name.decode()}")
    
    def calculate_pvalues(
        self,
        alt_base_counts: np.ndarray,
        depths: np.ndarray,
        alt_cell_counts: np.ndarray,
        total_cells: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate beta-binomial p-values for both base counts and cell counts.
        
        Args:
            alt_base_counts: Number of reads supporting alternative allele
            depths: Total read depth at each position
            alt_cell_counts: Number of cells supporting alternative allele
            total_cells: Total number of cells at each position
        
        Returns:
            Tuple of (base_count_pvalues, cell_count_pvalues)
        """
        bc_pvalues = betabinom_sf_batch(
            alt_base_counts, depths, self.alpha1, self.beta1, self.backend
        )
        
        cc_pvalues = betabinom_sf_batch(
            alt_cell_counts, total_cells, self.alpha2, self.beta2, self.backend
        )
        
        return bc_pvalues, cc_pvalues
    
    def calculate_noise_pvalues(
        self,
        sum_alt_base_counts: np.ndarray,
        sum_depths: np.ndarray,
        sum_alt_cell_counts: np.ndarray,
        sum_total_cells: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate noise p-values for site-level filtering.
        """
        return self.calculate_pvalues(
            sum_alt_base_counts, sum_depths,
            sum_alt_cell_counts, sum_total_cells
        )
    
    def is_significant(
        self,
        bc_pvalues: np.ndarray,
        cc_pvalues: np.ndarray,
        threshold: float = 0.001,
    ) -> np.ndarray:
        """
        Determine which positions have significant variants.
        Both base count and cell count p-values must be below threshold.
        """
        return (bc_pvalues < threshold) & (cc_pvalues < threshold)


def benchmark_backends(n_tests: int = 10000, max_k: int = 100):
    """Benchmark different backends."""
    import time
    
    np.random.seed(42)
    k = np.random.randint(1, max_k, n_tests).astype(np.float64)
    n = np.random.randint(k.astype(int) + 1, max_k * 2, n_tests).astype(np.float64)
    alpha, beta = 0.26, 173.9
    
    results = {}
    
    print(f"\nBenchmarking beta-binomial SF with {n_tests:,} tests...")
    print("-" * 50)
    
    t0 = time.perf_counter()
    scipy_result = np.array([scipy_betabinom.sf(ki - 0.1, int(ni), alpha, beta) 
                             for ki, ni in zip(k, n)])
    t1 = time.perf_counter()
    results["scipy (row-by-row)"] = t1 - t0
    print(f"scipy (row-by-row):  {t1-t0:.4f}s")
    
    t0 = time.perf_counter()
    numpy_result = betabinom_sf_batch_numpy(k, n, alpha, beta)
    t1 = time.perf_counter()
    results["numpy (batch)"] = t1 - t0
    print(f"numpy (batch):       {t1-t0:.4f}s  (speedup: {results['scipy (row-by-row)']/(t1-t0):.1f}x)")
    
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        cupy_result = betabinom_sf_batch_cupy(k, n, alpha, beta)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        results["cupy (GPU)"] = t1 - t0
        print(f"cupy (GPU):          {t1-t0:.4f}s  (speedup: {results['scipy (row-by-row)']/(t1-t0):.1f}x)")
    
    if HAS_JAX:
        _ = betabinom_sf_batch_jax(k[:10], n[:10], alpha, beta)
        
        t0 = time.perf_counter()
        jax_result = betabinom_sf_batch_jax(k, n, alpha, beta)
        jax_result.block_until_ready()
        t1 = time.perf_counter()
        results["jax (JIT)"] = t1 - t0
        print(f"jax (JIT):           {t1-t0:.4f}s  (speedup: {results['scipy (row-by-row)']/(t1-t0):.1f}x)")
    
    print("-" * 50)
    print("\nValidation (first 5 values):")
    print(f"  scipy:  {scipy_result[:5]}")
    print(f"  numpy:  {numpy_result[:5]}")
    if HAS_CUPY:
        print(f"  cupy:   {cp.asnumpy(cupy_result)[:5]}")
    if HAS_JAX:
        print(f"  jax:    {np.asarray(jax_result)[:5]}")
    
    return results


if __name__ == "__main__":
    print("GPU Beta-Binomial Calculator")
    print("=" * 50)
    print(f"CuPy available: {HAS_CUPY}")
    print(f"JAX available:  {HAS_JAX}")
    print(f"Selected backend: {get_backend()}")
    
    benchmark_backends(n_tests=10000)

