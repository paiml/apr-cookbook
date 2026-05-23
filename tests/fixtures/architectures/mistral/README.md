# Mistral Synthetic Micro-Fixture

2-layer Mistral-style config used by `examples/inference/inference_mistral_smoke.rs`.

## Provenance
Hand-authored 2026-05-07. Reduced from upstream Mistral-7B config.json:
- `num_hidden_layers`: 32 → 2
- `hidden_size`: 4096 → 64
- `vocab_size`: 32000 → 256
- `sliding_window`: 4096 → 64 (proportional to hidden_size)

The `sliding_window` field is the architecturally distinguishing feature
of Mistral vs Llama — sliding-window attention with that local context
size. The recipe asserts its presence as a Mistral-vs-Llama discriminator.

## Why bundled, not generated
- `architecture-demos.md` non-goal: no HF Hub network calls at runtime.
- Fixture is < 1 KB; no Git LFS overhead.
- Deterministic across machines and CI runners.
