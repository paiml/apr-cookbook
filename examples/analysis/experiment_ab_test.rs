//! # Recipe: Experiment — A/B/C Test Across 3 Model Configurations
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr experiment --configs baseline,variant_a,variant_b --trials 500`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example experiment_ab_test` exits 0
//! 2. [x] `cargo test --example experiment_ab_test` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr experiment` in-process (no shell-out)
//! 10. [x] Unit tests cover ranking, ties, z-statistic
//!
//! ## Learning Objective
//! Runs three configurations through a synthetic reward simulator, ranks them
//! deterministically (mean, then name for ties) and computes a two-proportion
//! z-statistic between the best and baseline arms.
//!
//! ## Run Command
//! ```bash
//! cargo run --example experiment_ab_test
//! ```
//!
//! ## References
//! - Kohavi, R. & Longbotham, R. (2017). *Online Controlled Experiments and A/B Testing*. DOI: 10.1007/978-1-4899-7687-1_891

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone)]
struct ArmSpec {
    name: String,
    p_success: f64,
}

#[derive(Debug, Clone)]
struct ArmStats {
    name: String,
    trials: usize,
    successes: usize,
    mean: f64,
}

fn run_arm(rng: &mut impl Rng, arm: &ArmSpec, trials: usize) -> ArmStats {
    let successes = (0..trials).filter(|_| rng.gen_bool(arm.p_success)).count();
    let mean = if trials == 0 {
        0.0
    } else {
        successes as f64 / trials as f64
    };
    ArmStats {
        name: arm.name.clone(),
        trials,
        successes,
        mean,
    }
}

fn rank_arms(mut arms: Vec<ArmStats>) -> Vec<ArmStats> {
    // Sort descending by mean; break ties by name ascending for determinism.
    arms.sort_by(|a, b| {
        b.mean
            .partial_cmp(&a.mean)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    arms
}

/// Two-proportion z-statistic (pooled variance).
fn two_proportion_z(a: &ArmStats, b: &ArmStats) -> f64 {
    if a.trials == 0 || b.trials == 0 {
        return 0.0;
    }
    let p1 = a.mean;
    let p2 = b.mean;
    let n1 = a.trials as f64;
    let n2 = b.trials as f64;
    let pooled = (a.successes + b.successes) as f64 / (n1 + n2);
    let var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2);
    if var <= 0.0 {
        0.0
    } else {
        (p1 - p2) / var.sqrt()
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("experiment_ab_test")?;
    println!("=== Recipe: {} ===", ctx.name());

    let arms = vec![
        ArmSpec {
            name: "baseline".into(),
            p_success: 0.62,
        },
        ArmSpec {
            name: "variant_a".into(),
            p_success: 0.68,
        },
        ArmSpec {
            name: "variant_b".into(),
            p_success: 0.71,
        },
    ];

    let trials_per_arm = 500;
    let mut stats = Vec::new();
    for arm in &arms {
        stats.push(run_arm(ctx.rng(), arm, trials_per_arm));
    }

    let ranked = rank_arms(stats.clone());
    println!("\n--- Arm Ranking ---");
    for (i, s) in ranked.iter().enumerate() {
        println!(
            "{}. {:<12} mean={:.4} ({:>3}/{:<3})",
            i + 1,
            s.name,
            s.mean,
            s.successes,
            s.trials
        );
    }

    // z-stat between winner and baseline.
    let baseline = stats.iter().find(|s| s.name == "baseline");
    let winner = ranked.first();
    let z = match (winner, baseline) {
        (Some(w), Some(b)) => two_proportion_z(w, b),
        _ => 0.0,
    };
    println!("\nWinner vs baseline z-statistic: {:.4}", z);
    println!(
        "|z| > 1.96 => statistically significant at α=0.05: {}",
        z.abs() > 1.96
    );

    let report = json!({
        "recipe": ctx.name(),
        "trials_per_arm": trials_per_arm,
        "arms": ranked.iter().map(|s| json!({
            "name": s.name,
            "trials": s.trials,
            "successes": s.successes,
            "mean": s.mean,
        })).collect::<Vec<_>>(),
        "z_winner_vs_baseline": z,
        "significant_0_05": z.abs() > 1.96,
    });
    let out = ctx.path("ab-test.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.record_float_metric("winner_mean", ranked.first().map_or(0.0, |s| s.mean));
    ctx.record_float_metric("z_statistic", z);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(name: &str, n: usize, k: usize) -> ArmStats {
        ArmStats {
            name: name.into(),
            trials: n,
            successes: k,
            mean: if n == 0 { 0.0 } else { k as f64 / n as f64 },
        }
    }

    #[test]
    fn ranks_by_mean_descending() {
        let arms = vec![stat("a", 100, 50), stat("b", 100, 80), stat("c", 100, 60)];
        let r = rank_arms(arms);
        assert_eq!(r[0].name, "b");
        assert_eq!(r[1].name, "c");
        assert_eq!(r[2].name, "a");
    }

    #[test]
    fn ties_break_by_name() {
        let arms = vec![stat("b", 10, 5), stat("a", 10, 5)];
        let r = rank_arms(arms);
        assert_eq!(r[0].name, "a");
    }

    #[test]
    fn z_stat_zero_for_equal_means() {
        let a = stat("a", 100, 50);
        let b = stat("b", 100, 50);
        let z = two_proportion_z(&a, &b);
        assert!(z.abs() < 1e-12);
    }

    #[test]
    fn z_stat_positive_when_a_better() {
        let a = stat("a", 100, 80);
        let b = stat("b", 100, 40);
        let z = two_proportion_z(&a, &b);
        assert!(z > 3.0);
    }

    #[test]
    fn z_stat_zero_trials_is_zero() {
        let a = stat("a", 0, 0);
        let b = stat("b", 100, 50);
        assert_eq!(two_proportion_z(&a, &b), 0.0);
    }
}
