//! # Contracts-Macros Severity Score Normalize
//!
//! Normalize severity scores from arbitrary scales to a 0..100
//! canonical range. Returns normalized scores and the conversion
//! factor used.
//!
//! Demonstrates the **CMM.189** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CVSS 3.1 base-score normalization §4.1; OWASP risk
//!  rating methodology.
//!
//! Run with: cargo run --example contracts_macros_severity_score_normalize
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NormalizeVerdict {
    Ok {
        normalized: Vec<u32>,
        scale_factor_x100: u32,
    },
    InvalidConfig,
}

pub fn normalize(scores: &[u32], source_max: u32) -> NormalizeVerdict {
    if scores.is_empty() || source_max == 0 || source_max > 1000 {
        return NormalizeVerdict::InvalidConfig;
    }
    for s in scores {
        if *s > source_max {
            return NormalizeVerdict::InvalidConfig;
        }
    }
    let factor = 100.0 / source_max as f64;
    let normalized: Vec<u32> = scores.iter().map(|s| (*s as f64 * factor) as u32).collect();
    NormalizeVerdict::Ok {
        normalized,
        scale_factor_x100: (factor * 100.0) as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_severity_score_normalize")?;

    println!("from /10: {:?}", normalize(&[1, 5, 10], 10));
    println!("from /5: {:?}", normalize(&[1, 3, 5], 5));
    println!("invalid: {:?}", normalize(&[], 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(normalize(&[], 10), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(normalize(&[5], 0), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_max_too_high() {
        assert_eq!(normalize(&[5], 10_000), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_score_over_max() {
        assert_eq!(normalize(&[15], 10), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn from_10_max_to_100_correct() {
        let v = normalize(&[1, 5, 10], 10);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized, vec![10, 50, 100]);
        }
    }

    #[test]
    fn from_5_max_to_100_correct() {
        let v = normalize(&[5], 5);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized, vec![100]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = normalize(&[5], 10);
        let r2 = normalize(&[5], 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn scale_factor_correct() {
        let v = normalize(&[5], 10);
        if let NormalizeVerdict::Ok {
            scale_factor_x100, ..
        } = v
        {
            // 100/10 = 10.0 → 1000 (×100)
            assert_eq!(scale_factor_x100, 1000);
        }
    }

    #[test]
    fn zero_score_normalizes_zero() {
        let v = normalize(&[0], 10);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized, vec![0]);
        }
    }

    #[test]
    fn many_scores_handled() {
        let scores: Vec<u32> = (0..30).collect();
        let v = normalize(&scores, 100);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized.len(), 30);
        }
    }

    #[test]
    fn max_score_normalizes_to_100() {
        let v = normalize(&[7], 7);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0], 100);
        }
    }

    #[test]
    fn unit_scale_passthrough() {
        let v = normalize(&[42, 80], 100);
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized, vec![42, 80]);
        }
    }
}
