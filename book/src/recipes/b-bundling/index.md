# Category B: Binary Bundling

Embed ML models directly into Rust binaries for zero-dependency deployment.

## Recipes

| Recipe | Description | Status |
|--------|-------------|--------|
| [Bundle Static Model](./bundle-static.md) | Embed model with `include_bytes!()` | Verified |
| [Bundle Quantized Model](./bundle-quantized.md) | Reduce model size with quantization | Verified |
| [Bundle Encrypted Model](./bundle-encrypted.md) | Protect model weights | Verified |
| [Static Binary Embedding](./static-binary.md) | Full static linking | Verified |
| [Q4 Quantization](./quantized-q4.md) | 4-bit quantization | Verified |
| [Signed Models](./signed.md) | Cryptographic signing | Verified |
| [Lambda Package](./lambda-package.md) | AWS Lambda deployment | Verified |

## Learning Objectives

- Embed models using `include_bytes!()` macro
- Reduce binary size with quantization
- Protect intellectual property with encryption
- Create single-binary deployments

## Toyota Way: Muda (Waste Elimination)

Bundling eliminates external dependencies, reducing deployment complexity and potential failure points.
