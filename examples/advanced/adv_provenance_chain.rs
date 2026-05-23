//! # Advanced Provenance Chain Replay
//!
//! Audit a model's lineage: foundation → fine-tune → distill → quantize.
//! Each step records (parent_hash, operation, output_hash). Replayer
//! verifies the chain is uninterrupted.
//!
//! Demonstrates the **ADV.26** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA Provenance Specification + git commit graph.
//!
//! Run with: cargo run --example adv_provenance_chain
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageStep {
    pub parent_hash: String,
    pub operation: String,
    pub output_hash: String,
}

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok { depth: u32 },
    BrokenAt { step_index: usize, expected: String },
    EmptyChain,
    DuplicateOutput { hash: String },
}

pub fn replay(initial_hash: &str, steps: &[LineageStep]) -> ChainVerdict {
    if steps.is_empty() {
        return ChainVerdict::EmptyChain;
    }
    let mut current = initial_hash.to_string();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    seen.insert(current.clone());
    for (i, step) in steps.iter().enumerate() {
        if step.parent_hash != current {
            return ChainVerdict::BrokenAt {
                step_index: i,
                expected: current,
            };
        }
        if !seen.insert(step.output_hash.clone()) {
            return ChainVerdict::DuplicateOutput {
                hash: step.output_hash.clone(),
            };
        }
        current = step.output_hash.clone();
    }
    ChainVerdict::Ok {
        depth: steps.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_provenance_chain")?;

    let steps = vec![
        LineageStep {
            parent_hash: "h0".to_string(),
            operation: "finetune".to_string(),
            output_hash: "h1".to_string(),
        },
        LineageStep {
            parent_hash: "h1".to_string(),
            operation: "distill".to_string(),
            output_hash: "h2".to_string(),
        },
        LineageStep {
            parent_hash: "h2".to_string(),
            operation: "quantize".to_string(),
            output_hash: "h3".to_string(),
        },
    ];
    println!("valid chain: {:?}", replay("h0", &steps));

    let broken = vec![
        LineageStep {
            parent_hash: "h0".to_string(),
            operation: "x".to_string(),
            output_hash: "h1".to_string(),
        },
        LineageStep {
            parent_hash: "wrong".to_string(),
            operation: "y".to_string(),
            output_hash: "h2".to_string(),
        },
    ];
    println!("broken: {:?}", replay("h0", &broken));
    println!("empty: {:?}", replay("h0", &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<LineageStep> {
        vec![
            LineageStep {
                parent_hash: "h0".to_string(),
                operation: "a".to_string(),
                output_hash: "h1".to_string(),
            },
            LineageStep {
                parent_hash: "h1".to_string(),
                operation: "b".to_string(),
                output_hash: "h2".to_string(),
            },
        ]
    }

    #[test]
    fn replayer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_chain_ok() {
        let v = replay("h0", &typical());
        if let ChainVerdict::Ok { depth } = v {
            assert_eq!(depth, 2);
        }
    }

    #[test]
    fn broken_at_first_step() {
        let v = replay("wrong", &typical());
        assert!(matches!(v, ChainVerdict::BrokenAt { step_index: 0, .. }));
    }

    #[test]
    fn broken_at_middle_step() {
        let mut steps = typical();
        steps[1].parent_hash = "wrong".to_string();
        let v = replay("h0", &steps);
        assert!(matches!(v, ChainVerdict::BrokenAt { step_index: 1, .. }));
    }

    #[test]
    fn empty_chain_rejected() {
        assert_eq!(replay("h0", &[]), ChainVerdict::EmptyChain);
    }

    #[test]
    fn duplicate_output_detected() {
        let steps = vec![
            LineageStep {
                parent_hash: "h0".to_string(),
                operation: "a".to_string(),
                output_hash: "h1".to_string(),
            },
            LineageStep {
                parent_hash: "h1".to_string(),
                operation: "b".to_string(),
                output_hash: "h0".to_string(), // cycle.
            },
        ];
        assert!(matches!(
            replay("h0", &steps),
            ChainVerdict::DuplicateOutput { .. }
        ));
    }

    #[test]
    fn long_chain_depth_correct() {
        let steps: Vec<LineageStep> = (0..100)
            .map(|i| LineageStep {
                parent_hash: format!("h{i}"),
                operation: "step".to_string(),
                output_hash: format!("h{}", i + 1),
            })
            .collect();
        if let ChainVerdict::Ok { depth } = replay("h0", &steps) {
            assert_eq!(depth, 100);
        }
    }

    #[test]
    fn single_step_works() {
        let steps = vec![LineageStep {
            parent_hash: "h0".to_string(),
            operation: "x".to_string(),
            output_hash: "h1".to_string(),
        }];
        assert!(matches!(
            replay("h0", &steps),
            ChainVerdict::Ok { depth: 1 }
        ));
    }

    #[test]
    fn broken_carries_expected_hash() {
        let mut steps = typical();
        steps[0].parent_hash = "wrong".to_string();
        if let ChainVerdict::BrokenAt { expected, .. } = replay("h0", &steps) {
            assert_eq!(expected, "h0");
        }
    }

    #[test]
    fn deterministic() {
        let a = replay("h0", &typical());
        let b = replay("h0", &typical());
        assert_eq!(a, b);
    }

    #[test]
    fn output_collision_with_initial_rejected() {
        let steps = vec![LineageStep {
            parent_hash: "h0".to_string(),
            operation: "x".to_string(),
            output_hash: "h0".to_string(),
        }];
        assert!(matches!(
            replay("h0", &steps),
            ChainVerdict::DuplicateOutput { .. }
        ));
    }
}
