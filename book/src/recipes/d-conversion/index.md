# Category D: Format Conversion

Convert between ML model formats.

## Recipes

| Recipe | Description | Status |
|--------|-------------|--------|
| [SafeTensors to APR](./safetensors-to-apr.md) | Import HuggingFace models | Verified |
| [APR to GGUF](./apr-to-gguf.md) | Export for llama.cpp | Verified |
| [GGUF to APR](./gguf-to-apr.md) | Import GGUF models | Verified |
| [Phi to APR](./phi-to-apr.md) | Convert Microsoft Phi models | Verified |
| [ONNX to APR](./onnx-to-apr.md) | Import ONNX models | Verified |

## Supported Formats

- **APR**: Native format, zero-copy loading
- **SafeTensors**: HuggingFace standard
- **GGUF**: llama.cpp format
- **ONNX**: Cross-platform interchange
