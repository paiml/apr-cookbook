//! # apr ollama --min-tokens-per-sec — Throughput Gate
//!
//! Streaming inference reports per-chunk timestamps. Computing the
//! average tokens/sec over an inference reveals when GPU is throttled
//! or when CPU offload kicks in. Gate fails CI when avg tps drops
//! below a per-model floor (e.g., 7B Q4 should sustain ≥ 30 tps on
//! a 4090). This recipe builds the gate.
//!
//! Demonstrates the **OLLAMA.6** recipe for PMAT-120 (apr ollama coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender OLLAMA-001 + llama.cpp benchmark conventions
//!
//! Run with: cargo run --example cli_ollama_token_rate_gate
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RateVerdict {
    Pass { tps: f64 },
    Fail { tps: f64, floor: f64 },
    InvalidStream,
}

pub fn tokens_per_sec(token_count: u64, elapsed_secs: f64) -> Option<f64> {
    if elapsed_secs <= 0.0 || !elapsed_secs.is_finite() {
        return None;
    }
    Some(token_count as f64 / elapsed_secs)
}

pub fn gate(token_count: u64, elapsed_secs: f64, floor_tps: f64) -> RateVerdict {
    if !floor_tps.is_finite() || floor_tps < 0.0 {
        return RateVerdict::InvalidStream;
    }
    let Some(tps) = tokens_per_sec(token_count, elapsed_secs) else {
        return RateVerdict::InvalidStream;
    };
    if tps >= floor_tps {
        RateVerdict::Pass { tps }
    } else {
        RateVerdict::Fail {
            tps,
            floor: floor_tps,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_token_rate_gate")?;

    let cases = [
        (3000u64, 100.0, 30.0),
        (1500, 100.0, 30.0),
        (3000, 0.0, 30.0),
        (3000, 100.0, -1.0),
    ];
    for (tok, secs, floor) in cases {
        println!(
            "tokens={tok} elapsed={secs}s floor={floor}  →  {:?}",
            gate(tok, secs, floor)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn meets_floor_passes() {
        let v = gate(3000, 100.0, 30.0);
        assert!(matches!(v, RateVerdict::Pass { .. }));
    }

    #[test]
    fn below_floor_fails() {
        let v = gate(1500, 100.0, 30.0);
        assert!(matches!(v, RateVerdict::Fail { .. }));
    }

    #[test]
    fn at_floor_passes() {
        // 30 tps exactly meets a 30 tps floor.
        let v = gate(30, 1.0, 30.0);
        assert!(matches!(v, RateVerdict::Pass { .. }));
    }

    #[test]
    fn zero_elapsed_invalid() {
        assert_eq!(gate(3000, 0.0, 30.0), RateVerdict::InvalidStream);
    }

    #[test]
    fn negative_elapsed_invalid() {
        assert_eq!(gate(3000, -1.0, 30.0), RateVerdict::InvalidStream);
    }

    #[test]
    fn nan_elapsed_invalid() {
        assert_eq!(gate(3000, f64::NAN, 30.0), RateVerdict::InvalidStream);
    }

    #[test]
    fn negative_floor_invalid() {
        assert_eq!(gate(3000, 100.0, -1.0), RateVerdict::InvalidStream);
    }

    #[test]
    fn tokens_per_sec_basic_math() {
        let tps = tokens_per_sec(100, 5.0).unwrap();
        assert!((tps - 20.0).abs() < 1e-9);
    }

    #[test]
    fn tokens_per_sec_zero_tokens_yields_zero() {
        let tps = tokens_per_sec(0, 5.0).unwrap();
        assert!(tps.abs() < 1e-9);
    }
}
