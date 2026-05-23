//! # apr rosetta diff-tensors — PAD-Token-Flood Signal
//!
//! `apr rosetta diff-tensors` exists primarily because of the GH-186
//! "PAD-token flood" failure: when the lm_head weight matrix is
//! transposed (in_dim ↔ out_dim swap), the model dispatches every output
//! to the same vocabulary id (the one at the linear-map zero index) —
//! typically the PAD token. This recipe builds the signal classifier that
//! turns a sample of generated token ids into a "looks like PAD flood"
//! verdict, with the flood threshold tuned to the GH-186 incident.
//!
//! Demonstrates the **ROSETTA-DIFF.3** recipe for PMAT-097 (apr rosetta diff-tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-186 + Hugging Face PAD token convention
//!
//! Run with: cargo run --example cli_rosetta_diff_tensors_pad_token_signal
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum FloodVerdict {
    Healthy {
        distinct_tokens: usize,
        total: usize,
    },
    LikelyPadFlood {
        flood_token: u32,
        flood_fraction: f64,
    },
    Empty,
}

const FLOOD_FRACTION_THRESHOLD: f64 = 0.95; // ≥95% same id

pub fn classify_pad_flood(generated_ids: &[u32]) -> FloodVerdict {
    if generated_ids.is_empty() {
        return FloodVerdict::Empty;
    }
    let mut max_id: u32 = generated_ids[0];
    let mut max_count: usize = 0;
    let total = generated_ids.len();
    let distinct = {
        let mut s = std::collections::HashSet::new();
        for id in generated_ids {
            s.insert(*id);
        }
        s.len()
    };
    // Single-pass mode finder.
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for id in generated_ids {
        let c = counts.entry(*id).or_insert(0);
        *c += 1;
        if *c > max_count {
            max_count = *c;
            max_id = *id;
        }
    }
    let frac = max_count as f64 / total as f64;
    if frac >= FLOOD_FRACTION_THRESHOLD {
        FloodVerdict::LikelyPadFlood {
            flood_token: max_id,
            flood_fraction: frac,
        }
    } else {
        FloodVerdict::Healthy {
            distinct_tokens: distinct,
            total,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_diff_tensors_pad_token_signal")?;

    // Healthy generation: many distinct tokens.
    let healthy = vec![5u32, 17, 23, 99, 42, 1, 2, 3, 4, 5, 17, 88];
    // PAD flood: nearly all same id.
    let flood = vec![0u32; 100]
        .iter()
        .copied()
        .chain([42u32, 17u32, 88u32])
        .collect::<Vec<_>>();

    println!("healthy: {:?}", classify_pad_flood(&healthy));
    println!("flood:   {:?}", classify_pad_flood(&flood));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_signal_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_returns_empty() {
        assert_eq!(classify_pad_flood(&[]), FloodVerdict::Empty);
    }

    #[test]
    fn diverse_output_is_healthy() {
        let ids: Vec<u32> = (0..50).collect();
        match classify_pad_flood(&ids) {
            FloodVerdict::Healthy {
                distinct_tokens,
                total,
            } => {
                assert_eq!(distinct_tokens, 50);
                assert_eq!(total, 50);
            }
            v => panic!("expected Healthy, got {v:?}"),
        }
    }

    #[test]
    fn dominant_token_above_95pct_flagged_as_flood() {
        let mut ids = vec![0u32; 100];
        ids.extend([1, 2, 3]); // 100/103 ≈ 97% → flood
        match classify_pad_flood(&ids) {
            FloodVerdict::LikelyPadFlood {
                flood_token,
                flood_fraction,
            } => {
                assert_eq!(flood_token, 0);
                assert!(flood_fraction > 0.95);
            }
            v => panic!("expected LikelyPadFlood, got {v:?}"),
        }
    }

    #[test]
    fn boundary_at_exactly_95pct_is_flood() {
        // Conservative-fail at the threshold (matches contract).
        let mut ids = vec![0u32; 19]; // 19/20 = 95%
        ids.push(1);
        let v = classify_pad_flood(&ids);
        assert!(matches!(v, FloodVerdict::LikelyPadFlood { .. }));
    }

    #[test]
    fn boundary_below_95pct_is_healthy() {
        // 18/20 = 90% — below threshold.
        let mut ids = vec![0u32; 18];
        ids.extend([1, 2]);
        let v = classify_pad_flood(&ids);
        assert!(matches!(v, FloodVerdict::Healthy { .. }));
    }

    #[test]
    fn single_token_output_is_flood() {
        // Pathological case: only one token generated, 100% same id.
        let v = classify_pad_flood(&[42]);
        if let FloodVerdict::LikelyPadFlood { flood_token, .. } = v {
            assert_eq!(flood_token, 42);
        } else {
            panic!("single-token must be flood");
        }
    }
}
