//! # apr train sweep — Grid Config Generator
//!
//! `apr train sweep <BASE.yaml>` generates the cartesian product of
//! hyperparameter values declared in the base file. This recipe builds
//! the grid expander and asserts the contract: empty input yields one
//! config (just the base), single-key sweep yields N configs, multi-key
//! yields cartesian-product, deterministic ordering for diff-able CI.
//!
//! Demonstrates the **TRAIN.15** recipe for PMAT-106 (apr train sweep coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TRAIN-SWEEP-001 + grid-search convention
//!
//! Run with: cargo run --example cli_train_sweep_grid_generator
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepAxis {
    pub key: String,
    pub values: Vec<String>,
}

pub fn cartesian(axes: &[SweepAxis]) -> Vec<BTreeMap<String, String>> {
    if axes.is_empty() {
        return vec![BTreeMap::new()];
    }
    let mut configs: Vec<BTreeMap<String, String>> = vec![BTreeMap::new()];
    for axis in axes {
        let mut next = Vec::new();
        for prefix in &configs {
            for value in &axis.values {
                let mut new = prefix.clone();
                new.insert(axis.key.clone(), value.clone());
                next.push(new);
            }
        }
        configs = next;
    }
    configs
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_sweep_grid_generator")?;

    let axes = vec![
        SweepAxis {
            key: "lr".into(),
            values: vec!["1e-4".into(), "5e-5".into(), "2e-5".into()],
        },
        SweepAxis {
            key: "batch_size".into(),
            values: vec!["32".into(), "64".into()],
        },
    ];

    let configs = cartesian(&axes);
    println!("Generated {} configs:", configs.len());
    for (i, c) in configs.iter().enumerate() {
        println!("  config {i}: {c:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_axes_yields_single_empty_config() {
        let configs = cartesian(&[]);
        assert_eq!(configs.len(), 1);
        assert!(configs[0].is_empty());
    }

    #[test]
    fn single_axis_with_n_values_yields_n_configs() {
        let axes = vec![SweepAxis {
            key: "lr".into(),
            values: vec!["1e-4".into(), "5e-5".into(), "2e-5".into()],
        }];
        let configs = cartesian(&axes);
        assert_eq!(configs.len(), 3);
    }

    #[test]
    fn two_axes_yield_cartesian_product() {
        // 3 × 2 = 6 configs.
        let axes = vec![
            SweepAxis {
                key: "lr".into(),
                values: vec!["1e-4".into(), "5e-5".into(), "2e-5".into()],
            },
            SweepAxis {
                key: "batch_size".into(),
                values: vec!["32".into(), "64".into()],
            },
        ];
        let configs = cartesian(&axes);
        assert_eq!(configs.len(), 6);
    }

    #[test]
    fn axis_with_empty_values_yields_empty_config_set() {
        // If any axis has 0 values, cartesian product is 0 configs.
        let axes = vec![SweepAxis {
            key: "lr".into(),
            values: vec![],
        }];
        let configs = cartesian(&axes);
        assert!(configs.is_empty());
    }

    #[test]
    fn each_config_includes_one_value_per_axis() {
        let axes = vec![
            SweepAxis {
                key: "a".into(),
                values: vec!["1".into(), "2".into()],
            },
            SweepAxis {
                key: "b".into(),
                values: vec!["x".into(), "y".into()],
            },
        ];
        let configs = cartesian(&axes);
        for c in &configs {
            assert_eq!(c.len(), 2);
            assert!(c.contains_key("a"));
            assert!(c.contains_key("b"));
        }
    }

    #[test]
    fn output_includes_all_combinations() {
        let axes = vec![
            SweepAxis {
                key: "a".into(),
                values: vec!["1".into(), "2".into()],
            },
            SweepAxis {
                key: "b".into(),
                values: vec!["x".into(), "y".into()],
            },
        ];
        let configs = cartesian(&axes);
        let pairs: std::collections::HashSet<(String, String)> = configs
            .iter()
            .map(|c| (c["a"].clone(), c["b"].clone()))
            .collect();
        let expected: std::collections::HashSet<_> = vec![
            ("1".to_string(), "x".to_string()),
            ("1".into(), "y".into()),
            ("2".into(), "x".into()),
            ("2".into(), "y".into()),
        ]
        .into_iter()
        .collect();
        assert_eq!(pairs, expected);
    }
}
