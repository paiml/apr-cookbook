//! # apr finetune — `--method` Auto-Picker
//!
//! `apr finetune --method auto` picks LoRA / QLoRA / full based on
//! available VRAM and model size. This recipe builds the picker as a
//! pure function and asserts the contract: known explicit methods pass
//! through, `auto` resolves deterministically per (model_size_b, vram_gb)
//! pair, unknown methods reject.
//!
//! Demonstrates the **FINETUNE.4** recipe for PMAT-104 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-244 + LoRA/QLoRA memory characteristics
//!
//! Run with: cargo run --example cli_finetune_method_picker
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Full,
    Lora,
    QLora,
}

impl Method {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Method::Full),
            "lora" => Some(Method::Lora),
            "qlora" => Some(Method::QLora),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Resolved(Method),
    UnknownMethod(String),
}

pub fn pick_method(method_arg: &str, model_size_b: f64, vram_gb: f64) -> PickerVerdict {
    if method_arg != "auto" {
        return match Method::from_str_strict(method_arg) {
            Some(m) => PickerVerdict::Resolved(m),
            None => PickerVerdict::UnknownMethod(method_arg.into()),
        };
    }
    // Heuristic: rough memory budget = model_size_b * 4 (bf16) * 2 (gradients + optimizer state)
    // = 8x model_size for full FT. LoRA needs ~1.5x. QLoRA needs ~0.6x.
    let full_budget_gb = model_size_b * 8.0;
    let lora_budget_gb = model_size_b * 1.5;
    if vram_gb >= full_budget_gb {
        PickerVerdict::Resolved(Method::Full)
    } else if vram_gb >= lora_budget_gb {
        PickerVerdict::Resolved(Method::Lora)
    } else {
        PickerVerdict::Resolved(Method::QLora)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_method_picker")?;

    let cases = [
        ("explicit lora", "lora", 7.0, 16.0),
        ("auto on H100/7B", "auto", 7.0, 80.0),
        ("auto on 4090/7B", "auto", 7.0, 24.0),
        ("auto on 4060Ti/7B", "auto", 7.0, 16.0),
        ("auto on 70B", "auto", 70.0, 24.0),
        ("typo", "loraa", 7.0, 24.0),
    ];

    for (label, method, size, vram) in cases {
        println!("{label:>22}  →  {:?}", pick_method(method, size, vram));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn explicit_lora_passes_through() {
        assert_eq!(
            pick_method("lora", 7.0, 16.0),
            PickerVerdict::Resolved(Method::Lora)
        );
    }

    #[test]
    fn explicit_full_passes_through() {
        assert_eq!(
            pick_method("full", 0.5, 100.0),
            PickerVerdict::Resolved(Method::Full)
        );
    }

    #[test]
    fn auto_picks_full_on_large_vram() {
        // 7B × 8 = 56GB budget; 80GB H100 → Full FT.
        assert_eq!(
            pick_method("auto", 7.0, 80.0),
            PickerVerdict::Resolved(Method::Full)
        );
    }

    #[test]
    fn auto_picks_lora_on_medium_vram() {
        // 7B × 1.5 = 10.5GB budget; 24GB 4090 → LoRA (full needs 56GB).
        assert_eq!(
            pick_method("auto", 7.0, 24.0),
            PickerVerdict::Resolved(Method::Lora)
        );
    }

    #[test]
    fn auto_picks_qlora_on_tight_vram() {
        // 70B × 1.5 = 105GB > 24GB → QLoRA.
        assert_eq!(
            pick_method("auto", 70.0, 24.0),
            PickerVerdict::Resolved(Method::QLora)
        );
    }

    #[test]
    fn unknown_method_rejected() {
        assert!(matches!(
            pick_method("loraa", 7.0, 24.0),
            PickerVerdict::UnknownMethod(_)
        ));
    }

    #[test]
    fn empty_method_rejected_as_unknown() {
        assert!(matches!(
            pick_method("", 7.0, 24.0),
            PickerVerdict::UnknownMethod(_)
        ));
    }
}
