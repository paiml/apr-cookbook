# CGP — `aprender-cgp` Compute-GPU-Profile

Recipes for [`aprender-cgp`](https://crates.io/crates/aprender-cgp) v0.31.2 — cross-backend kernel profiler. Run the same kernel through scalar / SIMD / wgpu / CUDA paths, get a unified report with throughput, latency, energy estimate, and roofline-model placement.

Closes the **≥3 recipes per sister crate** requirement from [expand-cookbooks/subcrate-coverage.md](../../specifications/expand-cookbooks/subcrate-coverage.md).

## Recipes

| # | Recipe | What |
|---|--------|------|
| CGP.1 | [`cgp_regression_detector_baseline_vs_current`](https://github.com/paiml/apr-cookbook/blob/main/examples/cgp/cgp_regression_detector_baseline_vs_current.rs) | Bootstrap CI regression detector (Hoefler & Belli SC'15); 10% slowdown → `Verdict::Regression` |
| CGP.2 | [`cgp_roofline_classify_kernel`](https://github.com/paiml/apr-cookbook/blob/main/examples/cgp/cgp_roofline_classify_kernel.rs) | Synthetic RTX 4090 roofline; classify low-AI/high-AI kernels as memory-bound vs compute-bound |
| CGP.3 | [`cgp_roofline_ridge_point_per_precision`](https://github.com/paiml/apr-cookbook/blob/main/examples/cgp/cgp_roofline_ridge_point_per_precision.rs) | Ridge points across FP32/TF32/BF16/FP16/INT8; INT8 ridge = 2× FP16 ridge |

## API surface exercised

- `cgp::analysis::regression::{RegressionDetector, Verdict}` — bootstrap CIs
- `cgp::analysis::roofline::{RooflineModel, Precision, MemoryLevel, Bound}`

GPU backends (`wgpu`, `cuda`) are gated behind cargo features and skipped on the cookbook's CI runner; scalar baseline always exercised.

## Provenance

Added during PMAT-083 (expand-cookbooks initiative, v6.1.0).
