# Convert with Quantization

**CLI Equivalent:** `apr convert --quantize`

## What This Demonstrates

Converts a model between formats while simultaneously applying quantization (e.g., FP32 to INT8 or Q4_0). This avoids a separate quantization pass and ensures the target format receives already-quantized tensors.

## Run

```bash
cargo run --example format_convert_quantize
```

## Key APIs

- `ConvertConfig::new(source_fmt, target_fmt)` — Configure a format conversion
- `.with_quantization(Quantization::Int8)` — Apply quantization during conversion
- `.with_calibration_data(dataset)` — Provide calibration data for quantization-aware conversion
- `Converter::run(input, output, config)` — Execute the combined convert-and-quantize pipeline

## Source

[`examples/format/format_convert_quantize.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_convert_quantize.rs)
