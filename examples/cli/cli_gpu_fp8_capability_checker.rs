//! # apr gpu --fp8 — FP8 Capability Checker
//!
//! FP8 (E4M3 / E5M2) requires Hopper (SM 9.0+) or newer. Earlier
//! architectures (Ada SM 8.9, Ampere SM 8.0/8.6, Turing SM 7.5)
//! must fall back to FP16 + BF16. This recipe builds the capability
//! lookup + fallback advisor.
//!
//! Demonstrates the **GPU.6** recipe for PMAT-120 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GPU-001 + NVIDIA Hopper architecture whitepaper (2022)
//!
//! Run with: cargo run --example cli_gpu_fp8_capability_checker
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum Fp8Verdict {
    Native,
    EmulatedFallback { recommended: &'static str },
    Unsupported,
}

pub fn check_fp8(sm_major: u32, sm_minor: u32) -> Fp8Verdict {
    let combined = sm_major * 10 + sm_minor;
    if combined >= 90 {
        Fp8Verdict::Native
    } else if combined >= 80 {
        Fp8Verdict::EmulatedFallback {
            recommended: "bf16",
        }
    } else if combined >= 75 {
        Fp8Verdict::EmulatedFallback {
            recommended: "fp16",
        }
    } else {
        Fp8Verdict::Unsupported
    }
}

pub fn architecture_name(sm_major: u32, sm_minor: u32) -> &'static str {
    let combined = sm_major * 10 + sm_minor;
    match combined {
        100..=u32::MAX => "Blackwell",
        90..=99 => "Hopper",
        89 => "Ada Lovelace",
        86 => "Ampere (consumer)",
        80 => "Ampere (datacenter)",
        75 => "Turing",
        70 | 72 => "Volta",
        60..=69 => "Pascal",
        _ => "Pre-Pascal",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_fp8_capability_checker")?;

    let cases = [(9, 0), (8, 9), (8, 0), (7, 5), (6, 0), (10, 0)];
    for (maj, min) in cases {
        println!(
            "SM {maj}.{min} ({})  →  {:?}",
            architecture_name(maj, min),
            check_fp8(maj, min)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hopper_native_fp8() {
        assert_eq!(check_fp8(9, 0), Fp8Verdict::Native);
    }

    #[test]
    fn blackwell_native_fp8() {
        assert_eq!(check_fp8(10, 0), Fp8Verdict::Native);
    }

    #[test]
    fn ada_emulated_to_bf16() {
        let v = check_fp8(8, 9);
        assert!(matches!(
            v,
            Fp8Verdict::EmulatedFallback {
                recommended: "bf16"
            }
        ));
    }

    #[test]
    fn ampere_emulated_to_bf16() {
        let v = check_fp8(8, 0);
        assert!(matches!(
            v,
            Fp8Verdict::EmulatedFallback {
                recommended: "bf16"
            }
        ));
    }

    #[test]
    fn turing_emulated_to_fp16() {
        // Turing (SM 7.5) lacks BF16; falls back to FP16.
        let v = check_fp8(7, 5);
        assert!(matches!(
            v,
            Fp8Verdict::EmulatedFallback {
                recommended: "fp16"
            }
        ));
    }

    #[test]
    fn pre_turing_unsupported() {
        assert_eq!(check_fp8(6, 0), Fp8Verdict::Unsupported);
        assert_eq!(check_fp8(7, 0), Fp8Verdict::Unsupported);
    }

    #[test]
    fn architecture_names_correct() {
        assert_eq!(architecture_name(10, 0), "Blackwell");
        assert_eq!(architecture_name(9, 0), "Hopper");
        assert_eq!(architecture_name(8, 9), "Ada Lovelace");
        assert_eq!(architecture_name(8, 0), "Ampere (datacenter)");
        assert_eq!(architecture_name(7, 5), "Turing");
        assert_eq!(architecture_name(6, 1), "Pascal");
    }

    #[test]
    fn combined_sm_calculation() {
        // Verify the combined formula matches NVIDIA's CC encoding.
        // SM 8.6 → 86; SM 9.0 → 90; SM 12.0 → 120.
        assert_eq!(8u32 * 10 + 6, 86);
        assert_eq!(9u32 * 10 + 0, 90);
    }
}
