//! # Optimize AdamW vs Lion Optimizer Picker
//!
//! AdamW: Adam with decoupled weight decay; well-tested, more memory
//! (2 moments per param). Lion: sign-based momentum; ~2× more memory-
//! efficient than AdamW; competitive on language/vision but needs
//! 3-10× lower LR. This recipe picks based on memory budget +
//! sensitivity tolerance.
//!
//! Demonstrates the **OPT.27** recipe for PMAT-131 (optimize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Chen et al. (2023). Symbolic Discovery of Optimization Algorithms (Lion).
//!
//! Run with: cargo run --example optimize_adamw_vs_lion_picker
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    AdamW,
    Lion,
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Ok {
        optimizer: Optimizer,
        recommended_lr_factor: f64,
    },
    InvalidParamCount,
}

const ADAMW_BYTES_PER_PARAM: u32 = 8; // 2 floats × 4 bytes (m, v at FP32)
const LION_BYTES_PER_PARAM: u32 = 4; // 1 float × 4 bytes (m only)

pub fn pick(num_params: u64, memory_budget_bytes: u64) -> PickerVerdict {
    if num_params == 0 {
        return PickerVerdict::InvalidParamCount;
    }
    let adamw_need = num_params * u64::from(ADAMW_BYTES_PER_PARAM);
    let lion_need = num_params * u64::from(LION_BYTES_PER_PARAM);
    if adamw_need <= memory_budget_bytes {
        PickerVerdict::Ok {
            optimizer: Optimizer::AdamW,
            recommended_lr_factor: 1.0,
        }
    } else if lion_need <= memory_budget_bytes {
        PickerVerdict::Ok {
            optimizer: Optimizer::Lion,
            // Lion needs lower LR — typical 3-10× scale-down.
            recommended_lr_factor: 0.1,
        }
    } else {
        // Neither fits — fall back to Lion as smaller; caller deals with OOM risk.
        PickerVerdict::Ok {
            optimizer: Optimizer::Lion,
            recommended_lr_factor: 0.1,
        }
    }
}

pub fn optimizer_state_bytes(optimizer: Optimizer, num_params: u64) -> u64 {
    let per_param = match optimizer {
        Optimizer::AdamW => ADAMW_BYTES_PER_PARAM,
        Optimizer::Lion => LION_BYTES_PER_PARAM,
    };
    num_params * u64::from(per_param)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("optimize_adamw_vs_lion_picker")?;

    let params_7b: u64 = 7_000_000_000;
    for budget_gb in [80u64, 40, 20] {
        let bytes = budget_gb * 1024 * 1024 * 1024;
        println!(
            "7B model, {budget_gb} GB budget  →  {:?}",
            pick(params_7b, bytes)
        );
    }
    println!(
        "AdamW state for 7B: {} GB",
        optimizer_state_bytes(Optimizer::AdamW, params_7b) / (1024 * 1024 * 1024)
    );
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
    fn ample_budget_picks_adamw() {
        // 1M params × 8 bytes = 8 MB; budget 1 GB.
        let v = pick(1_000_000, 1_000_000_000);
        assert!(matches!(
            v,
            PickerVerdict::Ok {
                optimizer: Optimizer::AdamW,
                ..
            }
        ));
    }

    #[test]
    fn tight_budget_picks_lion() {
        // 1M params × 8 = 8 MB AdamW; × 4 = 4 MB Lion. Budget 6 MB.
        let v = pick(1_000_000, 6_000_000);
        assert!(matches!(
            v,
            PickerVerdict::Ok {
                optimizer: Optimizer::Lion,
                ..
            }
        ));
    }

    #[test]
    fn extreme_budget_falls_back_to_lion() {
        // Neither fits; smaller still picked.
        let v = pick(1_000_000, 1_000);
        assert!(matches!(
            v,
            PickerVerdict::Ok {
                optimizer: Optimizer::Lion,
                ..
            }
        ));
    }

    #[test]
    fn lion_uses_lower_lr_factor() {
        let v = pick(1_000_000, 6_000_000);
        if let PickerVerdict::Ok {
            recommended_lr_factor,
            ..
        } = v
        {
            assert!(recommended_lr_factor < 1.0);
        }
    }

    #[test]
    fn adamw_uses_unit_lr_factor() {
        let v = pick(1_000_000, 1_000_000_000);
        if let PickerVerdict::Ok {
            recommended_lr_factor,
            ..
        } = v
        {
            assert_eq!(recommended_lr_factor, 1.0);
        }
    }

    #[test]
    fn zero_params_invalid() {
        assert_eq!(pick(0, 1_000_000), PickerVerdict::InvalidParamCount);
    }

    #[test]
    fn state_size_basic_math() {
        // 1M params × 8 bytes (AdamW).
        assert_eq!(
            optimizer_state_bytes(Optimizer::AdamW, 1_000_000),
            8_000_000
        );
        assert_eq!(optimizer_state_bytes(Optimizer::Lion, 1_000_000), 4_000_000);
    }

    #[test]
    fn lion_state_half_of_adamw() {
        let n = 7_000_000_000u64;
        assert_eq!(
            optimizer_state_bytes(Optimizer::AdamW, n),
            optimizer_state_bytes(Optimizer::Lion, n) * 2
        );
    }

    #[test]
    fn boundary_at_adamw_budget_picks_adamw() {
        // 1M × 8 = 8M; budget 8M exactly.
        let v = pick(1_000_000, 8_000_000);
        assert!(matches!(
            v,
            PickerVerdict::Ok {
                optimizer: Optimizer::AdamW,
                ..
            }
        ));
    }
}
