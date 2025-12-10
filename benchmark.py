#!/usr/bin/env python3
"""
SComatic Pipeline Benchmarking Script

Benchmarks each step of the SComatic pipeline:
- Step 1: Split BAM into cell-type-specific BAMs
- Step 2: Collect base count information  
- Step 3: Merge base count matrices
- Step 4.1: Apply hard filters and Beta-binomial tests
- Step 4.2: Apply additional filters (RNA editing, PoN, clustering)
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    step_name: str
    duration_seconds: float
    success: bool
    output_files: list[str]
    error_message: str = ""


class SCoMaticBenchmark:
    def __init__(
        self,
        base_dir: Path,
        bam_file: Path,
        meta_file: Path,
        ref_file: Path,
        output_dir: Path,
        sample_id: str = "benchmark",
        chrom: str = "chr10",
        nprocs: int = 1,
    ):
        self.base_dir = base_dir
        self.scripts_dir = base_dir / "scripts"
        self.bam_file = bam_file
        self.meta_file = meta_file
        self.ref_file = ref_file
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.chrom = chrom
        self.nprocs = nprocs
        self.results: list[BenchmarkResult] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step1_dir = self.output_dir / "step1_split_bam"
        self.step2_dir = self.output_dir / "step2_base_counts"
        self.step3_dir = self.output_dir / "step3_merged"
        self.step4_dir = self.output_dir / "step4_calling"
        self.temp_dir = self.output_dir / "temp"

    def _run_python_script(self, script_path: Path, args: list[str]) -> tuple[float, bool, str]:
        cmd = [sys.executable, str(script_path)] + args
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start
        success = result.returncode == 0
        error = result.stderr if not success else ""
        if not success:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return duration, success, error

    def step1_split_bam(self) -> BenchmarkResult:
        print("\n" + "=" * 60)
        print("STEP 1: Split BAM into cell-type-specific BAMs")
        print("=" * 60)

        self.step1_dir.mkdir(parents=True, exist_ok=True)

        script = self.scripts_dir / "SplitBam" / "SplitBamCellTypes.py"
        args = [
            "--bam", str(self.bam_file),
            "--meta", str(self.meta_file),
            "--id", self.sample_id,
            "--outdir", str(self.step1_dir),
            "--min_MQ", "255",
        ]

        duration, success, error = self._run_python_script(script, args)
        output_files = list(self.step1_dir.glob("*.bam"))

        result = BenchmarkResult(
            step_name="Step 1: Split BAM",
            duration_seconds=duration,
            success=success,
            output_files=[str(f) for f in output_files],
            error_message=error,
        )
        self.results.append(result)
        self._print_result(result)
        return result

    def step2_base_cell_counter(self) -> BenchmarkResult:
        print("\n" + "=" * 60)
        print("STEP 2: Collect base count information")
        print("=" * 60)

        self.step2_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        cell_type_bams = list(self.step1_dir.glob("*.bam"))
        cell_type_bams = [f for f in cell_type_bams if not f.name.endswith(".bam.bai")]

        if not cell_type_bams:
            result = BenchmarkResult(
                step_name="Step 2: Base Cell Counter",
                duration_seconds=0,
                success=False,
                output_files=[],
                error_message="No cell-type BAM files found from Step 1",
            )
            self.results.append(result)
            self._print_result(result)
            return result

        script = self.scripts_dir / "BaseCellCounter" / "BaseCellCounter.py"
        total_duration = 0.0
        all_success = True
        all_errors = []

        for bam in cell_type_bams:
            print(f"  Processing: {bam.name}")
            args = [
                "--bam", str(bam),
                "--ref", str(self.ref_file),
                "--chrom", self.chrom,
                "--out_folder", str(self.step2_dir),
                "--nprocs", str(self.nprocs),
                "--tmp_dir", str(self.temp_dir),
                "--min_dp", "5",
                "--min_cc", "5",
            ]

            duration, success, error = self._run_python_script(script, args)
            total_duration += duration
            if not success:
                all_success = False
                all_errors.append(f"{bam.name}: {error}")

        output_files = list(self.step2_dir.glob("*.tsv"))
        result = BenchmarkResult(
            step_name="Step 2: Base Cell Counter",
            duration_seconds=total_duration,
            success=all_success,
            output_files=[str(f) for f in output_files],
            error_message="\n".join(all_errors),
        )
        self.results.append(result)
        self._print_result(result)
        return result

    def step3_merge_counts(self) -> BenchmarkResult:
        print("\n" + "=" * 60)
        print("STEP 3: Merge base count matrices")
        print("=" * 60)

        self.step3_dir.mkdir(parents=True, exist_ok=True)
        merged_file = self.step3_dir / f"{self.sample_id}.BaseCellCounts.AllCellTypes.tsv"

        script = self.scripts_dir / "MergeCounts" / "MergeBaseCellCounts.py"
        args = [
            "--tsv_folder", str(self.step2_dir),
            "--outfile", str(merged_file),
        ]

        duration, success, error = self._run_python_script(script, args)
        output_files = [merged_file] if merged_file.exists() else []

        result = BenchmarkResult(
            step_name="Step 3: Merge Counts",
            duration_seconds=duration,
            success=success,
            output_files=[str(f) for f in output_files],
            error_message=error,
        )
        self.results.append(result)
        self._print_result(result)
        return result

    def step4_1_calling(self) -> BenchmarkResult:
        print("\n" + "=" * 60)
        print("STEP 4.1: Variant calling (Beta-binomial tests)")
        print("=" * 60)

        self.step4_dir.mkdir(parents=True, exist_ok=True)
        merged_file = self.step3_dir / f"{self.sample_id}.BaseCellCounts.AllCellTypes.tsv"
        output_prefix = self.step4_dir / self.sample_id

        script = self.scripts_dir / "BaseCellCalling" / "BaseCellCalling.step1.py"
        args = [
            "--infile", str(merged_file),
            "--outfile", str(output_prefix),
            "--ref", str(self.ref_file),
        ]

        duration, success, error = self._run_python_script(script, args)
        output_files = list(self.step4_dir.glob("*.step1.tsv"))

        result = BenchmarkResult(
            step_name="Step 4.1: Variant Calling (Beta-binomial)",
            duration_seconds=duration,
            success=success,
            output_files=[str(f) for f in output_files],
            error_message=error,
        )
        self.results.append(result)
        self._print_result(result)
        return result

    def step4_2_filtering(self, editing_file: Path | None = None, pon_file: Path | None = None) -> BenchmarkResult:
        print("\n" + "=" * 60)
        print("STEP 4.2: Additional filtering (RNA editing, PoN)")
        print("=" * 60)

        step1_output = self.step4_dir / f"{self.sample_id}.calling.step1.tsv"
        output_prefix = self.step4_dir / self.sample_id

        script = self.scripts_dir / "BaseCellCalling" / "BaseCellCalling.step2.py"
        args = [
            "--infile", str(step1_output),
            "--outfile", str(output_prefix),
        ]

        if editing_file and editing_file.exists():
            args.extend(["--editing", str(editing_file)])
        if pon_file and pon_file.exists():
            args.extend(["--pon", str(pon_file)])

        duration, success, error = self._run_python_script(script, args)
        output_files = list(self.step4_dir.glob("*.step2.tsv"))

        result = BenchmarkResult(
            step_name="Step 4.2: Additional Filtering",
            duration_seconds=duration,
            success=success,
            output_files=[str(f) for f in output_files],
            error_message=error,
        )
        self.results.append(result)
        self._print_result(result)
        return result

    def _print_result(self, result: BenchmarkResult):
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"\n  {status}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")
        print(f"  Output files: {len(result.output_files)}")
        for f in result.output_files[:5]:
            print(f"    - {Path(f).name}")
        if len(result.output_files) > 5:
            print(f"    ... and {len(result.output_files) - 5} more")
        if result.error_message:
            print(f"  Error: {result.error_message[:200]}...")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        total_time = sum(r.duration_seconds for r in self.results)
        successful = sum(1 for r in self.results if r.success)

        print(f"\n{'Step':<45} {'Time (s)':<12} {'Status':<10}")
        print("-" * 67)

        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"{r.step_name:<45} {r.duration_seconds:<12.2f} {status:<10}")

        print("-" * 67)
        print(f"{'TOTAL':<45} {total_time:<12.2f} {successful}/{len(self.results)} passed")
        print()

        return self.results

    def save_results(self, output_file: Path):
        total_time = sum(r.duration_seconds for r in self.results)
        successful = sum(1 for r in self.results if r.success)
        timestamp = datetime.now().isoformat()

        lines = [
            "=" * 70,
            "SCOMATIC BENCHMARK RESULTS",
            "=" * 70,
            "",
            f"Timestamp: {timestamp}",
            f"BAM file: {self.bam_file}",
            f"Metadata: {self.meta_file}",
            f"Reference: {self.ref_file}",
            f"Chromosome: {self.chrom}",
            f"Processes: {self.nprocs}",
            f"Output dir: {self.output_dir}",
            "",
            "=" * 70,
            "STEP TIMINGS",
            "=" * 70,
            "",
            f"{'Step':<45} {'Time (s)':<12} {'Status':<10}",
            "-" * 67,
        ]

        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            lines.append(f"{r.step_name:<45} {r.duration_seconds:<12.2f} {status:<10}")

        lines.extend([
            "-" * 67,
            f"{'TOTAL':<45} {total_time:<12.2f} {successful}/{len(self.results)} passed",
            "",
            "=" * 70,
            "DETAILED RESULTS",
            "=" * 70,
        ])

        for r in self.results:
            lines.extend([
                "",
                f"[{r.step_name}]",
                f"  Duration: {r.duration_seconds:.4f} seconds",
                f"  Status: {'SUCCESS' if r.success else 'FAILED'}",
                f"  Output files ({len(r.output_files)}):",
            ])
            for f in r.output_files:
                lines.append(f"    - {Path(f).name}")
            if r.error_message:
                lines.append(f"  Error: {r.error_message[:500]}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(lines) + "\n")
        print(f"\nResults saved to: {output_file}")

        json_file = output_file.with_suffix(".json")
        json_data = {
            "timestamp": timestamp,
            "config": {
                "bam_file": str(self.bam_file),
                "meta_file": str(self.meta_file),
                "ref_file": str(self.ref_file),
                "chrom": self.chrom,
                "nprocs": self.nprocs,
                "output_dir": str(self.output_dir),
            },
            "total_time_seconds": total_time,
            "steps_passed": successful,
            "steps_total": len(self.results),
            "results": [asdict(r) for r in self.results],
        }
        json_file.write_text(json.dumps(json_data, indent=2))
        print(f"JSON results saved to: {json_file}")

    def run_full_pipeline(
        self,
        editing_file: Path | None = None,
        pon_file: Path | None = None,
        results_file: Path | None = None,
    ) -> list[BenchmarkResult]:
        print("\n" + "#" * 60)
        print("  SCOMATIC PIPELINE BENCHMARK")
        print("#" * 60)
        print(f"\nConfiguration:")
        print(f"  BAM file: {self.bam_file}")
        print(f"  Metadata: {self.meta_file}")
        print(f"  Reference: {self.ref_file}")
        print(f"  Chromosome: {self.chrom}")
        print(f"  Processes: {self.nprocs}")
        print(f"  Output: {self.output_dir}")

        self.step1_split_bam()
        self.step2_base_cell_counter()
        self.step3_merge_counts()
        self.step4_1_calling()
        self.step4_2_filtering(editing_file, pon_file)

        self.print_summary()

        if results_file is None:
            results_file = self.output_dir / "benchmark_results.txt"
        self.save_results(results_file)

        return self.results


def main():
    parser = argparse.ArgumentParser(description="SComatic Pipeline Benchmarking")
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).parent, help="SComatic base directory")
    parser.add_argument("--bam", type=Path, help="Input BAM file")
    parser.add_argument("--meta", type=Path, help="Cell barcode metadata file")
    parser.add_argument("--ref", type=Path, help="Reference genome FASTA file")
    parser.add_argument("--output", type=Path, default=Path("benchmark_output"), help="Output directory")
    parser.add_argument("--sample_id", type=str, default="benchmark", help="Sample ID prefix")
    parser.add_argument("--chrom", type=str, default="chr10", help="Chromosome to analyze")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of processes for parallel steps")
    parser.add_argument("--editing", type=Path, help="RNA editing sites file (optional)")
    parser.add_argument("--pon", type=Path, help="Panel of Normals file (optional)")
    parser.add_argument("--use_example", action="store_true", help="Use example data")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before running")
    parser.add_argument("--results_file", type=Path, help="Output file for benchmark results")

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()

    if args.use_example:
        example_dir = base_dir / "example_data"
        bam_file = example_dir / "Example.scrnaseq.bam"
        meta_file = example_dir / "Example.cell_barcode_annotations.tsv"
        ref_file = example_dir / "chr10.fa"
        editing_file = base_dir / "RNAediting" / "AllEditingSites.hg38.txt"
        pon_file = base_dir / "PoNs" / "PoN.scRNAseq.hg38.tsv"
    else:
        bam_file = args.bam
        meta_file = args.meta
        ref_file = args.ref
        editing_file = args.editing
        pon_file = args.pon

        if not all([bam_file, meta_file, ref_file]):
            print("ERROR: --bam, --meta, and --ref are required unless --use_example is specified")
            sys.exit(1)

    output_dir = args.output.resolve()

    if args.clean and output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    benchmark = SCoMaticBenchmark(
        base_dir=base_dir,
        bam_file=bam_file,
        meta_file=meta_file,
        ref_file=ref_file,
        output_dir=output_dir,
        sample_id=args.sample_id,
        chrom=args.chrom,
        nprocs=args.nprocs,
    )

    results_file = args.results_file
    if results_file:
        results_file = results_file.resolve()

    results = benchmark.run_full_pipeline(
        editing_file=editing_file,
        pon_file=pon_file,
        results_file=results_file,
    )

    all_passed = all(r.success for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

