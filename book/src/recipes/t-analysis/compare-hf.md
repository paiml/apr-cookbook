# Compare HuggingFace

Bit-for-bit tensor comparison between a local APR model and HuggingFace SafeTensors weights. Maps HF naming conventions to APR naming, then computes per-tensor metrics: max absolute error, mean absolute error, cosine similarity, and L2 distance.

## CLI Equivalent
```bash
apr compare_hf model.apr --repo my-org/my-model --threshold 1e-5
```

## Key Concepts
- Tensor name mapping between APR and HuggingFace conventions
- Per-tensor numerical comparison (max error, cosine similarity, L2)
- Pass/fail gating per tensor and overall

## Run
```bash
cargo run --example analysis_compare_hf
```

## Source
[`examples/analysis/analysis_compare_hf/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_compare_hf/main.rs)
