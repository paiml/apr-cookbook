# Category U: Format

Format recipes demonstrate the APR format ecosystem: importing models from external hubs, exporting to interoperable formats (SafeTensors, GGUF), cross-format conversion via the Rosetta engine, quantized conversion, publishing, and batch operations. These mirror the `apr import`, `apr export`, `apr rosetta`, `apr convert`, `apr publish`, and `apr pull` CLI subcommands.

## Recipes

| # | Recipe | CLI Equivalent | Description |
|---|--------|---------------|-------------|
| 1 | [Import from HuggingFace](import-hf.md) | `apr import hf://org/repo` | Download and convert a HuggingFace model to `.apr` |
| 2 | [Export to SafeTensors](export-safetensors.md) | `apr export --format safetensors` | Serialize an `.apr` model to SafeTensors format |
| 3 | [Export to GGUF](export-gguf.md) | `apr export --format gguf` | Serialize an `.apr` model to GGUF for llama.cpp |
| 4 | [Rosetta Convert](rosetta-convert.md) | `apr rosetta convert` | Cross-format conversion via the Rosetta engine |
| 5 | [Rosetta Chain](rosetta-chain.md) | `apr rosetta chain` | Multi-step conversion chain (e.g., ONNX -> APR -> GGUF) |
| 6 | [Rosetta Verify](rosetta-verify.md) | `apr rosetta verify` | Round-trip verification of format fidelity |
| 7 | [Convert with Quantization](convert-quantize.md) | `apr convert --quantize` | Convert between formats with quantization applied |
| 8 | [Publish to HuggingFace](publish.md) | `apr publish` | Push an `.apr` model to a HuggingFace repository |
| 9 | [Pull and Cache](pull-cache.md) | `apr pull` | Download models with local cache management |
| 10 | [Batch Multi-Format Export](batch-export.md) | `apr export --batch` | Export a model to multiple formats in one pass |
