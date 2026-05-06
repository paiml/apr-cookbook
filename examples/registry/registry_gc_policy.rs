//! # Registry Garbage Collection Policy
//!
//! Registry GC removes unreferenced model versions to reclaim space.
//! Policy: keep all aliased versions; keep N most-recent unaliased
//! versions; keep all versions newer than `min_age_days`. This recipe
//! builds the per-version eligibility check.
//!
//! Demonstrates the **REG.10** recipe for PMAT-129 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender REG-001 + GitHub Container Registry GC docs.
//!
//! Run with: cargo run --example registry_gc_policy
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone)]
pub struct Version {
    pub id: String,
    pub age_days: u32,
    pub aliased: bool,
}

#[derive(Debug, PartialEq)]
pub enum GcAction {
    Keep,
    Delete,
}

#[derive(Debug, PartialEq)]
pub enum PolicyVerdict {
    Plan(Vec<(String, GcAction)>),
    InvalidConfig,
}

pub fn classify(versions: &[Version], keep_recent: u32, min_age_days: u32) -> PolicyVerdict {
    if keep_recent == 0 && min_age_days == 0 {
        // At least one safety net required.
        return PolicyVerdict::InvalidConfig;
    }
    let mut sorted: Vec<&Version> = versions.iter().collect();
    sorted.sort_by_key(|v| v.age_days);
    let mut kept_unaliased = 0u32;
    let mut plan: Vec<(String, GcAction)> = Vec::with_capacity(versions.len());
    for v in &sorted {
        let action = if v.aliased || v.age_days < min_age_days {
            GcAction::Keep
        } else if kept_unaliased < keep_recent {
            kept_unaliased += 1;
            GcAction::Keep
        } else {
            GcAction::Delete
        };
        plan.push((v.id.clone(), action));
    }
    PolicyVerdict::Plan(plan)
}

pub fn count_kept(plan: &[(String, GcAction)]) -> usize {
    plan.iter().filter(|(_, a)| *a == GcAction::Keep).count()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_gc_policy")?;

    let versions = vec![
        Version {
            id: "v1.0".into(),
            age_days: 200,
            aliased: true,
        },
        Version {
            id: "v1.1".into(),
            age_days: 100,
            aliased: false,
        },
        Version {
            id: "v1.2".into(),
            age_days: 50,
            aliased: false,
        },
        Version {
            id: "v1.3".into(),
            age_days: 10,
            aliased: false,
        },
        Version {
            id: "v1.4".into(),
            age_days: 1,
            aliased: false,
        },
    ];
    println!("plan: {:?}", classify(&versions, 2, 30));
    println!("invalid: {:?}", classify(&versions, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Version> {
        vec![
            Version {
                id: "v1.0".into(),
                age_days: 200,
                aliased: true,
            },
            Version {
                id: "v1.1".into(),
                age_days: 100,
                aliased: false,
            },
            Version {
                id: "v1.2".into(),
                age_days: 50,
                aliased: false,
            },
            Version {
                id: "v1.3".into(),
                age_days: 10,
                aliased: false,
            },
            Version {
                id: "v1.4".into(),
                age_days: 1,
                aliased: false,
            },
        ]
    }

    #[test]
    fn policy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn aliased_versions_always_kept() {
        if let PolicyVerdict::Plan(plan) = classify(&sample(), 0, 999) {
            let v0 = plan.iter().find(|(id, _)| id == "v1.0").unwrap();
            assert_eq!(v0.1, GcAction::Keep);
        }
    }

    #[test]
    fn young_versions_kept() {
        // min_age=30 → versions younger than 30 days kept.
        if let PolicyVerdict::Plan(plan) = classify(&sample(), 0, 30) {
            // v1.3 (10 days) and v1.4 (1 day) should be kept.
            for id in ["v1.3", "v1.4"] {
                let v = plan.iter().find(|(i, _)| i == id).unwrap();
                assert_eq!(v.1, GcAction::Keep);
            }
        }
    }

    #[test]
    fn keep_recent_keeps_n_most_recent_unaliased() {
        // Sort by age ascending: v1.4(1), v1.3(10), v1.2(50), v1.1(100).
        // keep_recent=2 → first 2 unaliased kept (v1.4, v1.3); v1.2, v1.1 deleted.
        if let PolicyVerdict::Plan(plan) = classify(&sample(), 2, 0) {
            assert_eq!(
                plan.iter().find(|(i, _)| i == "v1.4").unwrap().1,
                GcAction::Keep
            );
            assert_eq!(
                plan.iter().find(|(i, _)| i == "v1.3").unwrap().1,
                GcAction::Keep
            );
            assert_eq!(
                plan.iter().find(|(i, _)| i == "v1.2").unwrap().1,
                GcAction::Delete
            );
            assert_eq!(
                plan.iter().find(|(i, _)| i == "v1.1").unwrap().1,
                GcAction::Delete
            );
        }
    }

    #[test]
    fn invalid_config_no_safety_net() {
        // keep_recent=0 + min_age_days=0 → would delete everything; reject.
        assert_eq!(classify(&sample(), 0, 0), PolicyVerdict::InvalidConfig);
    }

    #[test]
    fn count_kept_helper_works() {
        if let PolicyVerdict::Plan(plan) = classify(&sample(), 2, 30) {
            // Aliased v1.0 (1) + young v1.4, v1.3 (2) + 2 most-recent unaliased
            // outside young window (v1.2, v1.1) = 5 kept; nothing deleted.
            assert_eq!(count_kept(&plan), 5);
        }
    }

    #[test]
    fn empty_versions_handled() {
        if let PolicyVerdict::Plan(plan) = classify(&[], 1, 0) {
            assert!(plan.is_empty());
        }
    }

    #[test]
    fn all_aliased_keep_all() {
        let v = vec![
            Version {
                id: "a".into(),
                age_days: 100,
                aliased: true,
            },
            Version {
                id: "b".into(),
                age_days: 200,
                aliased: true,
            },
        ];
        if let PolicyVerdict::Plan(plan) = classify(&v, 0, 1) {
            assert_eq!(count_kept(&plan), 2);
        }
    }

    #[test]
    fn high_min_age_keeps_all_unaliased_in_window() {
        let v = vec![
            Version {
                id: "old".into(),
                age_days: 1000,
                aliased: false,
            },
            Version {
                id: "new".into(),
                age_days: 1,
                aliased: false,
            },
        ];
        if let PolicyVerdict::Plan(plan) = classify(&v, 0, 500) {
            // Old is past window → delete (keep_recent=0).
            // New is within window → keep.
            assert_eq!(
                plan.iter().find(|(i, _)| i == "old").unwrap().1,
                GcAction::Delete
            );
            assert_eq!(
                plan.iter().find(|(i, _)| i == "new").unwrap().1,
                GcAction::Keep
            );
        }
    }
}
