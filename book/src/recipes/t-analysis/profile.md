# Roofline Profiling

**CLI Equivalent:** `apr profile model.apr --granular`

## What This Demonstrates

Performs roofline model analysis to classify each layer as compute-bound or memory-bound. Produces per-layer profiling with arithmetic intensity, an ASCII roofline chart, bottleneck identification, and optimization recommendations (quantize, prune, SIMD/GPU, distillation).

## Run

```bash
cargo run --example analysis_profile
```

## Key APIs

- `roofline_analysis(flops, bytes_accessed, &hw)` -- classify a layer as compute-bound or memory-bound
- `estimate_layer_profile(name, input_dim, output_dim, batch_size, &hw)` -- compute FLOPs, bytes, arithmetic intensity
- `HardwareSpec { peak_gflops, memory_bandwidth_gb_s }` -- target hardware specification with `ridge_point()`
- `generate_recommendations(&profiles, &hw)` -- prioritized optimization suggestions per layer
- `render_roofline_ascii(&profiles, &hw)` -- ASCII roofline chart with layer plot points

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_profile.rs}}
```

## Source

[`examples/analysis/analysis_profile.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_profile.rs)
