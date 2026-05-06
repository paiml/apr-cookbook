//! # apr rosetta validate-stats — Threshold Gate (z-score)
//!
//! `apr rosetta validate-stats <MODEL> --threshold <K>` flags every tensor
//! whose recorded statistic deviates by more than K standard deviations
//! from the reference fingerprint. Default K=3.0 catches genuine
//! corruption (≈3-sigma) without false-positive flagging on benign FP
//! round-trip noise.
//!
//! Demonstrates the **ROSETTA-VALIDATE.1** recipe for PMAT-097 (apr rosetta validate-stats coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-202 + 3-sigma rule
//!
//! Run with: cargo run --example cli_rosetta_validate_stats_threshold_gate
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct StatPair {
    pub tensor: String,
    pub observed: f64,
    pub expected: f64,
    pub expected_std: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviationFinding {
    pub tensor: String,
    pub z_score: f64,
}

pub fn z_score(observed: f64, expected: f64, std: f64) -> Option<f64> {
    if std <= 0.0 || !std.is_finite() {
        return None;
    }
    Some((observed - expected) / std)
}

pub fn flag_deviations(stats: &[StatPair], threshold: f64) -> Vec<DeviationFinding> {
    stats
        .iter()
        .filter_map(|s| {
            let z = z_score(s.observed, s.expected, s.expected_std)?;
            if z.abs() > threshold {
                Some(DeviationFinding {
                    tensor: s.tensor.clone(),
                    z_score: z,
                })
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_validate_stats_threshold_gate")?;

    let stats = vec![
        StatPair {
            tensor: "embed_tokens".into(),
            observed: 0.001,
            expected: 0.000,
            expected_std: 0.001,
        },
        StatPair {
            tensor: "layers.0.q_proj".into(),
            observed: 0.05,
            expected: 0.001,
            expected_std: 0.001, // z ≈ 49 — clearly bad
        },
        StatPair {
            tensor: "lm_head".into(),
            observed: -0.002,
            expected: 0.000,
            expected_std: 0.001, // z = -2 — within 3σ
        },
    ];

    for k in [3.0_f64, 1.5, 0.5] {
        let flagged = flag_deviations(&stats, k);
        println!("--threshold {k:.1}σ:  {} deviations", flagged.len());
        for f in &flagged {
            println!("  {} z={:.2}", f.tensor, f.z_score);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(t: &str, o: f64, e: f64, s: f64) -> StatPair {
        StatPair {
            tensor: t.into(),
            observed: o,
            expected: e,
            expected_std: s,
        }
    }

    #[test]
    fn threshold_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_threshold_not_flagged() {
        let stats = vec![sp("a", 0.001, 0.0, 0.001)]; // z = 1
        assert!(flag_deviations(&stats, 3.0).is_empty());
    }

    #[test]
    fn outside_threshold_flagged() {
        let stats = vec![sp("bad", 0.05, 0.001, 0.001)]; // z ≈ 49
        let f = flag_deviations(&stats, 3.0);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].tensor, "bad");
        assert!(f[0].z_score > 3.0);
    }

    #[test]
    fn negative_deviation_flagged_via_abs() {
        let stats = vec![sp("neg", -0.05, 0.0, 0.001)]; // z = -50
        let f = flag_deviations(&stats, 3.0);
        assert_eq!(f.len(), 1);
        assert!(f[0].z_score < 0.0);
    }

    #[test]
    fn zero_std_returns_none_not_inf() {
        // Avoid divide-by-zero — return None (caller emits "no fingerprint").
        assert_eq!(z_score(1.0, 0.0, 0.0), None);
    }

    #[test]
    fn negative_std_returns_none() {
        // Std must be non-negative. Negative means corrupted fingerprint.
        assert_eq!(z_score(1.0, 0.0, -0.5), None);
    }

    #[test]
    fn nan_std_returns_none() {
        assert_eq!(z_score(1.0, 0.0, f64::NAN), None);
    }

    #[test]
    fn boundary_at_exactly_threshold_not_flagged() {
        // Conservative-pass at the boundary (matches z-score statistical convention).
        let stats = vec![sp("boundary", 0.003, 0.0, 0.001)]; // z = 3.0 exactly
        assert!(flag_deviations(&stats, 3.0).is_empty());
    }
}
