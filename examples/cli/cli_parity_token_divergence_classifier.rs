//! # apr parity — Token Divergence Classifier
//!
//! `apr parity <FILE>` runs the same prompt on CPU and GPU paths and
//! compares output tokens. This recipe builds the divergence classifier
//! as a pure function over the (cpu_tokens, gpu_tokens) pair: ExactMatch,
//! FirstDivergenceAt(pos), or LengthMismatch. The position of first
//! divergence is critical for genchi-genbutsu debugging.
//!
//! Demonstrates the **PARITY.8** recipe for PMAT-111 (apr parity coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-232 + Toyota Way (genchi genbutsu)
//!
//! Run with: cargo run --example cli_parity_token_divergence_classifier
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DivergenceVerdict {
    ExactMatch,
    FirstDivergenceAt { position: usize, cpu: u32, gpu: u32 },
    LengthMismatch { cpu_len: usize, gpu_len: usize },
}

pub fn classify(cpu: &[u32], gpu: &[u32]) -> DivergenceVerdict {
    if cpu.len() != gpu.len() {
        return DivergenceVerdict::LengthMismatch {
            cpu_len: cpu.len(),
            gpu_len: gpu.len(),
        };
    }
    for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
        if c != g {
            return DivergenceVerdict::FirstDivergenceAt {
                position: i,
                cpu: *c,
                gpu: *g,
            };
        }
    }
    DivergenceVerdict::ExactMatch
}

pub fn exit_code(v: &DivergenceVerdict, assert_mode: bool) -> i32 {
    if matches!(v, DivergenceVerdict::ExactMatch) {
        return 0;
    }
    if assert_mode {
        65
    } else {
        0
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_parity_token_divergence_classifier")?;

    let cases = [
        ("identical", vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 4]),
        ("diverge at 2", vec![1, 2, 3, 4], vec![1, 2, 99, 4]),
        ("length mismatch", vec![1, 2, 3], vec![1, 2, 3, 4]),
    ];
    for (label, cpu, gpu) in cases {
        let v = classify(&cpu, &gpu);
        println!(
            "{label:>20}  →  {v:?}  exit(--assert)={}",
            exit_code(&v, true)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tokens_yield_exact_match() {
        let v = classify(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(v, DivergenceVerdict::ExactMatch);
    }

    #[test]
    fn first_divergence_position_reported() {
        let v = classify(&[1, 2, 3, 4], &[1, 2, 99, 4]);
        if let DivergenceVerdict::FirstDivergenceAt { position, cpu, gpu } = v {
            assert_eq!(position, 2);
            assert_eq!(cpu, 3);
            assert_eq!(gpu, 99);
        } else {
            panic!("expected FirstDivergenceAt");
        }
    }

    #[test]
    fn length_mismatch_shortcuts_per_token_check() {
        let v = classify(&[1, 2, 3], &[1, 2]);
        assert_eq!(
            v,
            DivergenceVerdict::LengthMismatch {
                cpu_len: 3,
                gpu_len: 2,
            }
        );
    }

    #[test]
    fn empty_inputs_yield_exact_match() {
        assert_eq!(classify(&[], &[]), DivergenceVerdict::ExactMatch);
    }

    #[test]
    fn assert_mode_returns_nonzero_on_divergence() {
        let v = DivergenceVerdict::FirstDivergenceAt {
            position: 0,
            cpu: 1,
            gpu: 2,
        };
        assert_ne!(exit_code(&v, true), 0);
        // Without --assert, divergence is reported but exit is 0.
        assert_eq!(exit_code(&v, false), 0);
    }

    #[test]
    fn exact_match_always_exits_zero() {
        let v = DivergenceVerdict::ExactMatch;
        assert_eq!(exit_code(&v, true), 0);
        assert_eq!(exit_code(&v, false), 0);
    }
}
