# Bundle with Quantization

Reduce model size 4-8x using quantization while maintaining accuracy.

## Quantization Levels

| Level | Bits | Size Reduction | Accuracy Loss |
|-------|------|----------------|---------------|
| F32 | 32 | 1x (baseline) | None |
| F16 | 16 | 2x | Minimal |
| Q8_0 | 8 | 4x | Low |
| Q4_0 | 4 | 8x | Moderate |

## Recipe

```rust
use apr_cookbook::convert::{AprConverter, TensorData, DataType};

fn quantize_model(weights: &[f32], target: DataType) -> Vec<u8> {
    match target {
        DataType::Q8_0 => {
            // Quantize to 8-bit
            weights.iter()
                .map(|&w| ((w.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8)
                .collect()
        }
        DataType::Q4_0 => {
            // Quantize to 4-bit (pack 2 values per byte)
            weights.chunks(2)
                .map(|pair| {
                    let a = ((pair[0].clamp(-1.0, 1.0) + 1.0) * 7.5) as u8;
                    let b = pair.get(1)
                        .map(|&w| ((w.clamp(-1.0, 1.0) + 1.0) * 7.5) as u8)
                        .unwrap_or(0);
                    (a << 4) | b
                })
                .collect()
        }
        _ => unimplemented!()
    }
}

fn main() {
    let mut converter = AprConverter::new();

    // Original F32 weights: 4 bytes per weight
    let f32_weights: Vec<f32> = vec![0.1, -0.5, 0.8, 0.0];

    // Quantize to Q8_0: 1 byte per weight (4x smaller)
    let q8_data = quantize_model(&f32_weights, DataType::Q8_0);

    converter.add_tensor(TensorData {
        name: "layer.weight".to_string(),
        shape: vec![2, 2],
        dtype: DataType::Q8_0,
        data: q8_data,
    });

    let bundle = converter.to_apr().unwrap();
    println!("Quantized bundle: {} bytes", bundle.len());
}
```

## Run the Example

```bash
cargo run --example bundle_quantized_model
```

## When to Quantize

| Model Size | Recommendation |
|------------|----------------|
| < 10 MB | F32 (no quantization) |
| 10-100 MB | F16 or Q8_0 |
| 100 MB - 1 GB | Q8_0 |
| > 1 GB | Q4_0 |
