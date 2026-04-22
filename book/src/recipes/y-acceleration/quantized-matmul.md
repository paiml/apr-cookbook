# Quantized Matrix Multiply

INT8 and INT4 quantized matrix multiplication that reduces memory bandwidth while preserving inference accuracy. Compares FP32 baseline, FP16 simulated, INT8 (scale + zero-point), and INT4 (packed 2-per-byte) approaches.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- INT8 and INT4 quantized matmul with scale/zero-point
- Memory bandwidth reduction (4-8x) via quantization
- Precision-accuracy tradeoff measurement

## Run
```bash
cargo run --example acceleration_quantized_matmul --release
```

## Source
[`examples/acceleration/acceleration_quantized_matmul/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/acceleration/acceleration_quantized_matmul/main.rs)
