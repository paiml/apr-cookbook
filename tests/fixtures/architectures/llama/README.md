# Llama Synthetic Micro-Fixture

2-layer Llama-style config used by `examples/inference/inference_llama_smoke.rs`.

## Provenance
Hand-authored 2026-05-07. Reduced from upstream Llama config.json:
- `num_hidden_layers`: 32 → 2
- `hidden_size`: 4096 → 64
- `vocab_size`: 32000 → 256
- All other fields preserved at their upstream semantics.

No weight file (`model.safetensors`) is committed — the smoke recipe
simulates a forward pass over the config without instantiating real
tensors. When upstream `aprender::rosetta::load_family('llama', ...)`
lands, regenerate this dir with synthetic seeded weights via
`scripts/architecture-demos-gen-fixture.py --family llama`.

## Why bundled, not generated
- `architecture-demos.md` non-goal: no HF Hub network calls at runtime.
- Fixture is < 1 KB; no Git LFS overhead.
- Deterministic across machines and CI runners.
