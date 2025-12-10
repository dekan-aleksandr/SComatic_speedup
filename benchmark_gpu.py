#!/usr/bin/env python3
"""GPU Benchmark for SComatic beta-binomial calculations (CuPy required)."""

import os
import sys
import time

import numpy as np
from scipy.stats import betabinom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts/BaseCellCalling"))

from gpu_betabinom import GPUBetaBinomCalculator, betabinom_sf_batch, HAS_CUPY

if not HAS_CUPY:
    from gpu_betabinom import CUPY_ERROR
    print(f"ERROR: CuPy (GPU) required but not available: {CUPY_ERROR}")
    print("Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)")
    sys.exit(1)

import cupy as cp

ALPHA1, BETA1 = 0.26, 173.9


def generate_test_data(n_tests: int) -> tuple:
    np.random.seed(42)
    depths = np.random.randint(10, 1000, size=n_tests).astype(np.float64)
    alt_counts = np.minimum(np.random.randint(1, 50, size=n_tests), depths).astype(np.float64)
    cell_totals = np.random.randint(5, 200, size=n_tests).astype(np.float64)
    alt_cells = np.minimum(np.random.randint(1, 20, size=n_tests), cell_totals).astype(np.float64)
    return alt_counts, depths, alt_cells, cell_totals


def benchmark_scipy(alt_counts, depths, alpha, beta):
    results = np.zeros(len(alt_counts))
    for i in range(len(alt_counts)):
        results[i] = betabinom.sf(alt_counts[i] - 0.1, depths[i], alpha, beta)
    return results


def run_benchmark(n_tests: int) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Benchmarking {n_tests:,} beta-binomial tests")
    print(f"{'=' * 60}")

    alt_counts, depths, _, _ = generate_test_data(n_tests)
    results = {"n_tests": n_tests}

    print("\n  [1/2] scipy row-by-row (baseline)...", end=" ", flush=True)
    start = time.perf_counter()
    scipy_results = benchmark_scipy(alt_counts, depths, ALPHA1, BETA1)
    scipy_time = time.perf_counter() - start
    results["scipy"] = scipy_time
    print(f"{scipy_time:.3f}s")

    print("  [2/2] CuPy GPU batch...", end=" ", flush=True)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    cupy_results = betabinom_sf_batch(alt_counts - 0.1, depths, ALPHA1, BETA1)
    cp.cuda.Stream.null.synchronize()
    cupy_time = time.perf_counter() - start
    results["cupy"] = cupy_time
    speedup = scipy_time / cupy_time
    print(f"{cupy_time:.3f}s ({speedup:.1f}x)")

    print("\n  Verifying correctness...")
    max_diff = np.max(np.abs(scipy_results - cupy_results))
    print(f"    scipy vs cupy: max diff = {max_diff:.2e} {'✓' if max_diff < 1e-6 else '✗'}")

    return results


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     SComatic GPU Benchmark: Beta-Binomial Calculations       ║
╚══════════════════════════════════════════════════════════════╝
""")

    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    test_sizes = [1_000, 10_000, 100_000, 500_000]

    all_results = []
    for n in test_sizes:
        results = run_benchmark(n)
        all_results.append(results)

    print("\n" + "=" * 60)
    print("                        SUMMARY")
    print("=" * 60)
    print(f"\n{'Tests':>12} | {'scipy':>10} | {'cupy':>10} | {'Speedup':>10}")
    print("-" * 60)

    for r in all_results:
        n = r["n_tests"]
        scipy_t = r["scipy"]
        cupy_t = r["cupy"]
        speedup = scipy_t / cupy_t
        print(f"{n:>12,} | {scipy_t:>9.3f}s | {cupy_t:>9.3f}s | {speedup:>9.1f}x")

    print("-" * 60)

    large_result = all_results[-1]
    scipy_t = large_result["scipy"]
    cupy_t = large_result["cupy"]
    speedup = scipy_t / cupy_t

    print(f"\nFor 500K tests: CuPy GPU is {speedup:.0f}x faster than scipy row-by-row")


if __name__ == "__main__":
    main()
