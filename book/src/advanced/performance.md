# Performance Optimization

Maximize inference performance with these techniques.

## Optimization Checklist

- [ ] Use `--release` builds
- [ ] Enable LTO (link-time optimization)
- [ ] Use `include_bytes!()` for models
- [ ] Enable SIMD via trueno
- [ ] Profile before optimizing

## Cargo.toml Settings

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"

[profile.release.package."*"]
opt-level = 3
```

## Memory Layout

Optimize tensor memory layout:

```rust
// Prefer contiguous arrays
let weights: Vec<f32> = vec![0.0; rows * cols];

// Access in row-major order
for i in 0..rows {
    for j in 0..cols {
        let idx = i * cols + j;
        process(weights[idx]);
    }
}
```

## Batch Processing

Process multiple inputs together:

```rust
fn infer_batch(model: &Model, inputs: &[Input]) -> Vec<Output> {
    // Single model load, multiple inferences
    inputs.iter()
        .map(|input| model.infer(input))
        .collect()
}
```

## Benchmarking

Use criterion for reliable benchmarks:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_inference(c: &mut Criterion) {
    let model = load_model();
    let input = create_test_input();

    c.bench_function("inference", |b| {
        b.iter(|| model.infer(&input))
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```
