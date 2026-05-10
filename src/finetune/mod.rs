//! Fine-tuning cookbook helpers shared across `examples/finetune/`.
//!
//! Each tier ships a thin wrapper module here; recipe files in
//! `examples/finetune/` import the shared helpers and only carry their
//! family-specific configuration. This keeps recipe files small and
//! lets the falsifier logic live in one well-tested place.
//!
//! See `docs/specifications/fine-tuning-cookbook.md` and `recipe-template.md`.

pub mod eval_primitives;
pub mod sft_minimal;
