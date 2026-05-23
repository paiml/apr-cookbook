//! # Contracts-Macros Recipe Priority Decay
//!
//! Apply exponential time-decay to recipe priorities. After
//! `current_day - last_touched_day` days, priority decays by
//! `decay_factor` per day. Returns final adjusted priorities.
//!
//! Demonstrates the **CMM.120** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TF-IDF time decay (Salton 1975); Hacker News rank
//!  algorithm (gravity factor).
//!
//! Run with: cargo run --example contracts_macros_recipe_priority_decay
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DecayVerdict {
    Ok {
        adjusted: Vec<(String, f64)>,
        max_priority: f64,
    },
    InvalidConfig,
}

pub fn decay(
    recipes: &[(&str, f64, u32)],
    current_day: u32,
    decay_factor_per_day: f64,
) -> DecayVerdict {
    if recipes.is_empty() || !(0.0..=1.0).contains(&decay_factor_per_day) {
        return DecayVerdict::InvalidConfig;
    }
    let mut adjusted: Vec<(String, f64)> = Vec::with_capacity(recipes.len());
    let mut max_priority: f64 = f64::MIN;
    for (name, base, last_touched) in recipes {
        if current_day < *last_touched {
            return DecayVerdict::InvalidConfig;
        }
        let age = current_day - *last_touched;
        let mult = decay_factor_per_day.powi(age as i32);
        let final_priority = base * mult;
        if final_priority > max_priority {
            max_priority = final_priority;
        }
        adjusted.push(((*name).to_string(), final_priority));
    }
    DecayVerdict::Ok {
        adjusted,
        max_priority,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_priority_decay")?;

    let recipes = [("fresh", 100.0, 5), ("stale", 100.0, 0)];
    println!("decay 5d, 0.95: {:?}", decay(&recipes, 10, 0.95));
    println!("invalid: {:?}", decay(&[], 10, 0.95));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decayer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_decay_factor_one() {
        let r = [("a", 100.0, 5)];
        let v = decay(&r, 10, 1.0);
        if let DecayVerdict::Ok { adjusted, .. } = v {
            assert!((adjusted[0].1 - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn full_decay_factor_zero() {
        let r = [("a", 100.0, 5)];
        let v = decay(&r, 10, 0.0);
        if let DecayVerdict::Ok { adjusted, .. } = v {
            assert!(adjusted[0].1 < 1e-9);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(decay(&[], 10, 0.5), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_decay_above_one() {
        let r = [("a", 100.0, 5)];
        assert_eq!(decay(&r, 10, 1.5), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_decay_negative() {
        let r = [("a", 100.0, 5)];
        assert_eq!(decay(&r, 10, -0.1), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn future_touch_rejected() {
        let r = [("a", 100.0, 20)];
        assert_eq!(decay(&r, 10, 0.5), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn fresh_higher_than_stale() {
        let r = [("fresh", 100.0, 9), ("stale", 100.0, 0)];
        let v = decay(&r, 10, 0.5);
        if let DecayVerdict::Ok { adjusted, .. } = v {
            assert!(adjusted[0].1 > adjusted[1].1);
        }
    }

    #[test]
    fn deterministic() {
        let r = [("a", 100.0, 5)];
        let r1 = decay(&r, 10, 0.95);
        let r2 = decay(&r, 10, 0.95);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_matches_input() {
        let r = [("a", 1.0, 0), ("b", 1.0, 0), ("c", 1.0, 0)];
        let v = decay(&r, 5, 0.5);
        if let DecayVerdict::Ok { adjusted, .. } = v {
            assert_eq!(adjusted.len(), 3);
        }
    }

    #[test]
    fn max_priority_correct() {
        let r = [("a", 50.0, 0), ("b", 100.0, 0)];
        let v = decay(&r, 0, 1.0);
        if let DecayVerdict::Ok { max_priority, .. } = v {
            assert!((max_priority - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn no_decay_when_age_zero() {
        let r = [("a", 100.0, 5)];
        let v = decay(&r, 5, 0.5);
        if let DecayVerdict::Ok { adjusted, .. } = v {
            assert!((adjusted[0].1 - 100.0).abs() < 1e-9);
        }
    }
}
