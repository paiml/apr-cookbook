# SIMD Acceleration

Leverage CPU SIMD instructions for faster inference via trueno integration.

## Supported Platforms

| Platform | Instructions | Auto-detected |
|----------|--------------|---------------|
| x86_64 | AVX2, AVX-512 | ✅ |
| ARM64 | NEON | ✅ |
| WASM | SIMD128 | ✅ |

## Recipe

```rust
use trueno::simd::{SimdBackend, detect_best_backend};

fn main() {
    // Auto-detect best SIMD backend
    let backend = detect_best_backend();
    println!("Using: {:?}", backend);

    // Matrix multiplication with SIMD
    let a = vec![1.0f32; 1024 * 1024];
    let b = vec![1.0f32; 1024 * 1024];

    let start = std::time::Instant::now();
    let c = backend.matmul(&a, &b, 1024, 1024, 1024);
    println!("MatMul: {:?}", start.elapsed());
}
```

## Run the Example

```bash
cargo run --example simd_matrix_operations --release
```

## Performance Comparison

| Operation | Scalar | AVX2 | Speedup |
|-----------|--------|------|---------|
| 1024x1024 MatMul | 2100ms | 85ms | 25x |
| Vector dot | 45μs | 3μs | 15x |
| Element-wise | 12μs | 1μs | 12x |

## Fallback Behavior

trueno automatically falls back to scalar operations when SIMD is unavailable:

```rust
// This works on all platforms
let result = backend.add(&a, &b);

// On x86_64 with AVX2: uses SIMD
// On older x86_64: uses scalar fallback
// On WASM: uses SIMD128 if available
```
