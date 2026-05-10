//! Fine-tuning cookbook helpers shared across `examples/finetune/`.
//!
//! Each tier ships a thin wrapper module here; recipe files in
//! `examples/finetune/` import the shared helpers and only carry their
//! family-specific configuration. This keeps recipe files small and
//! lets the falsifier logic live in one well-tested place.
//!
//! See `docs/specifications/fine-tuning-cookbook.md` and `recipe-template.md`.

pub mod adapter_merge;
pub mod anomaly_open_uncertainty;
pub mod calibration;
pub mod continued_pretrain;
pub mod encoders_optimizers;
pub mod eval_primitives;
pub mod hyperopt;
pub mod imbalance;
pub mod instruction_tuning;
pub mod lora;
pub mod memory_optimizers;
pub mod multimodal;
pub mod peft_variants;
pub mod preference;
pub mod qlora;
pub mod quantized_base;
pub mod rl_alignment;
pub mod sft_minimal;
pub mod smoke;
pub mod specialty;
pub mod tabular_classification;
pub mod tabular_regression;
pub mod tier3_closeout;
