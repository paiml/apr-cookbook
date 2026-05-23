//! # Monte-Carlo Wolfram Elementary CA Rule 30
//!
//! Run Wolfram's Rule 30 elementary cellular automaton on a 1D grid
//! for N steps starting from a single live cell. Returns final
//! live-cell count and total cells turned on across all steps.
//!
//! Demonstrates the **MC.192** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wolfram, A New Kind of Science (2002) ch. 2; Rule 30
//!  used as the random-bit stream for Mathematica's Random[].
//!
//! Run with: cargo run --example mc_elementary_ca_rule30
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CaVerdict {
    Ok {
        final_live: u32,
        total_live_steps: u32,
    },
    InvalidConfig,
}

pub fn simulate(width: u32, steps: u32) -> CaVerdict {
    if !(7..=1024).contains(&width) || !(1..=1024).contains(&steps) {
        return CaVerdict::InvalidConfig;
    }
    let w = width as usize;
    let mut row = vec![false; w];
    row[w / 2] = true; // single seed at center
    let mut total_live = 0u32;
    for _ in 0..steps {
        total_live += row.iter().filter(|c| **c).count() as u32;
        let mut next = vec![false; w];
        for i in 0..w {
            let l = if i == 0 { false } else { row[i - 1] };
            let c = row[i];
            let r = if i == w - 1 { false } else { row[i + 1] };
            // Rule 30: pattern (l, c, r) → output
            let pattern = (l as u8) << 2 | (c as u8) << 1 | r as u8;
            // Rule 30 binary: 00011110
            next[i] = (0b0001_1110u8 >> pattern) & 1 == 1;
        }
        row = next;
    }
    let final_live = row.iter().filter(|c| **c).count() as u32;
    CaVerdict::Ok {
        final_live,
        total_live_steps: total_live,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_elementary_ca_rule30")?;

    println!("w=51, 30 steps: {:?}", simulate(51, 30));
    println!("invalid: {:?}", simulate(3, 30));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_too_narrow_width() {
        assert_eq!(simulate(3, 30), CaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(51, 0), CaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(51, 30);
        let b = simulate(51, 30);
        assert_eq!(a, b);
    }

    #[test]
    fn final_live_at_least_one() {
        // Rule 30 is chaotic; almost always live cells remain.
        let v = simulate(51, 30);
        if let CaVerdict::Ok { final_live, .. } = v {
            assert!(final_live >= 1);
        }
    }

    #[test]
    fn total_live_grows_with_steps() {
        let short = simulate(51, 10);
        let long = simulate(51, 30);
        if let (
            CaVerdict::Ok {
                total_live_steps: s,
                ..
            },
            CaVerdict::Ok {
                total_live_steps: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn final_live_le_width() {
        let v = simulate(51, 10);
        if let CaVerdict::Ok { final_live, .. } = v {
            assert!(final_live <= 51);
        }
    }

    #[test]
    fn min_width_accepted() {
        let v = simulate(7, 1);
        assert!(matches!(v, CaVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(101, 100);
        assert!(matches!(v, CaVerdict::Ok { .. }));
    }

    #[test]
    fn rule30_seed_to_pattern() {
        // After 1 step from single seed at center, pattern ≈ 111 (Rule 30)
        let v = simulate(11, 2);
        if let CaVerdict::Ok { final_live, .. } = v {
            assert!(final_live >= 2);
        }
    }

    #[test]
    fn one_step_yields_growth() {
        let v = simulate(11, 1);
        if let CaVerdict::Ok {
            total_live_steps, ..
        } = v
        {
            // First step: one live cell counted.
            assert_eq!(total_live_steps, 1);
        }
    }

    #[test]
    fn larger_grid_more_total_live() {
        let small = simulate(11, 10);
        let large = simulate(101, 10);
        if let (
            CaVerdict::Ok {
                total_live_steps: s,
                ..
            },
            CaVerdict::Ok {
                total_live_steps: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }
}
