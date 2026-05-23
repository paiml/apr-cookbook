//! # apr train halving — Successive Halving Round Planner
//!
//! `apr train halving` runs successive-halving HPO (C-HPO-001): start
//! with N configs, eliminate worst half each round until 1 remains. This
//! recipe builds the round-by-round budget calculator and asserts the
//! contract: O(log N) rounds, total trials = N(1 + 1/2 + 1/4 + ...) ≈ 2N.
//!
//! Demonstrates the **TRAIN.16** recipe for PMAT-106 (apr train halving coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender C-HPO-001 + Jamieson & Talwalkar (2016) successive halving
//!
//! Run with: cargo run --example cli_train_halving_round_planner
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct Round {
    pub round_num: u32,
    pub configs_remaining: u32,
    pub budget_per_config: u32, // training steps per config this round
}

pub fn plan_halving(initial_configs: u32, max_steps_total: u32) -> Vec<Round> {
    if initial_configs == 0 || max_steps_total == 0 {
        return Vec::new();
    }
    // Compute number of rounds needed: log2(N).
    let n_rounds = (initial_configs as f64).log2().ceil() as u32 + 1;
    // Budget allocation: total budget / n_rounds, distributed equally.
    let per_round_budget = max_steps_total / n_rounds.max(1);
    let mut rounds = Vec::with_capacity(n_rounds as usize);
    let mut remaining = initial_configs;
    let mut round_num = 0;
    while remaining >= 1 {
        let per_config = per_round_budget / remaining.max(1);
        rounds.push(Round {
            round_num,
            configs_remaining: remaining,
            budget_per_config: per_config,
        });
        if remaining == 1 {
            break;
        }
        remaining = remaining.div_ceil(2); // halve, rounding up
        round_num += 1;
    }
    rounds
}

pub fn total_trials(rounds: &[Round]) -> u64 {
    rounds.iter().map(|r| u64::from(r.configs_remaining)).sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_halving_round_planner")?;

    for n in [4u32, 8, 16, 32, 100] {
        let rounds = plan_halving(n, 1_000);
        println!("N={n}, total_trials={}", total_trials(&rounds));
        for r in &rounds {
            println!(
                "  round {}: {} configs × {} steps",
                r.round_num, r.configs_remaining, r.budget_per_config
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_yields_no_rounds() {
        assert!(plan_halving(0, 1000).is_empty());
        assert!(plan_halving(10, 0).is_empty());
    }

    #[test]
    fn single_config_yields_single_round() {
        let r = plan_halving(1, 1000);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].configs_remaining, 1);
    }

    #[test]
    fn n_4_yields_3_rounds() {
        // 4 → 2 → 1 = 3 rounds.
        let r = plan_halving(4, 1000);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].configs_remaining, 4);
        assert_eq!(r[1].configs_remaining, 2);
        assert_eq!(r[2].configs_remaining, 1);
    }

    #[test]
    fn n_16_yields_5_rounds() {
        // 16 → 8 → 4 → 2 → 1 = 5 rounds.
        let r = plan_halving(16, 1000);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn rounds_strictly_decreasing() {
        let r = plan_halving(32, 1000);
        for w in r.windows(2) {
            assert!(w[1].configs_remaining <= w[0].configs_remaining);
        }
    }

    #[test]
    fn last_round_has_one_config() {
        for n in [1u32, 2, 4, 8, 100] {
            let r = plan_halving(n, 1000);
            assert_eq!(r.last().unwrap().configs_remaining, 1, "N={n}");
        }
    }

    #[test]
    fn total_trials_approximately_2n_for_large_n() {
        // SH-budget invariant: total ≈ N(1 + 1/2 + 1/4 + ...) ≈ 2N.
        let total = total_trials(&plan_halving(64, 100_000));
        // 64 + 32 + 16 + 8 + 4 + 2 + 1 = 127 ≈ 2×64-1.
        assert_eq!(total, 127);
    }
}
