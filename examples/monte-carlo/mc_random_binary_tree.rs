//! # Monte-Carlo Random Binary Tree Growth
//!
//! Sim insertion of N keys into a random binary search tree (insert
//! at random unoccupied leaf position). Returns mean tree height
//! and theoretical Catalan-bound height.
//!
//! Demonstrates the **MC.189** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Knuth TAOCP §6.2.2 random BSTs; Catalan numbers count
//!  binary tree shapes.
//!
//! Run with: cargo run --example mc_random_binary_tree
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TreeVerdict {
    Ok {
        mean_height: u32,
        theoretical_h_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_keys: u32, trials: u32, seed: u64) -> TreeVerdict {
    if n_keys < 5 || trials < 100 {
        return TreeVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_h: u64 = 0;
    for _ in 0..trials {
        // Insert n_keys random values into BST; track height.
        let mut tree: Vec<(i32, i32, u32)> = Vec::new(); // (left_idx, right_idx, depth)
        for _ in 0..n_keys {
            let key = (lcg(&mut state) as i32).abs();
            let mut idx = 0i32;
            let mut depth = 0u32;
            if tree.is_empty() {
                tree.push((-1, -1, 0));
                continue;
            }
            loop {
                depth += 1;
                let (left, right, _) = tree[idx as usize];
                let go_left = (key & 1) == 0;
                if go_left {
                    if left == -1 {
                        let new_idx = tree.len() as i32;
                        tree.push((-1, -1, depth));
                        tree[idx as usize].0 = new_idx;
                        break;
                    }
                    idx = left;
                } else {
                    if right == -1 {
                        let new_idx = tree.len() as i32;
                        tree.push((-1, -1, depth));
                        tree[idx as usize].1 = new_idx;
                        break;
                    }
                    idx = right;
                }
            }
        }
        let max_depth = tree.iter().map(|(_, _, d)| *d).max().unwrap_or(0);
        total_h += max_depth as u64;
    }
    let mean_h = (total_h / trials as u64) as u32;
    let theoretical = 4.31 * (n_keys as f64).ln(); // ~ E[H] ≈ 4.31 ln n
    TreeVerdict::Ok {
        mean_height: mean_h,
        theoretical_h_x100: (theoretical * 100.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_binary_tree")?;

    println!("n=100: {:?}", simulate(100, 500, 42));
    println!("invalid: {:?}", simulate(2, 100, 42));
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
    fn invalid_too_few_keys() {
        assert_eq!(simulate(2, 100, 42), TreeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 42), TreeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 100, 42);
        let b = simulate(20, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn larger_tree_taller() {
        let small = simulate(20, 200, 42);
        let large = simulate(200, 200, 42);
        if let (TreeVerdict::Ok { mean_height: s, .. }, TreeVerdict::Ok { mean_height: l, .. }) =
            (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn theoretical_returned() {
        let v = simulate(100, 100, 42);
        if let TreeVerdict::Ok {
            theoretical_h_x100, ..
        } = v
        {
            // 4.31 * ln(100) ≈ 19.85 → 1985
            assert!((1900..=2000).contains(&theoretical_h_x100));
        }
    }

    #[test]
    fn mean_height_at_least_one() {
        let v = simulate(20, 100, 42);
        if let TreeVerdict::Ok { mean_height, .. } = v {
            assert!(mean_height >= 1);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 100, 42);
        assert!(matches!(v, TreeVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(50, 5000, 42);
        assert!(matches!(v, TreeVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_produce_valid_outcomes() {
        let a = simulate(50, 100, 42);
        let b = simulate(50, 100, 999);
        assert!(matches!(a, TreeVerdict::Ok { .. }));
        assert!(matches!(b, TreeVerdict::Ok { .. }));
    }

    #[test]
    fn finite_outcomes() {
        let v = simulate(20, 100, 42);
        if let TreeVerdict::Ok {
            mean_height,
            theoretical_h_x100,
        } = v
        {
            assert!(mean_height < u32::MAX);
            assert!(theoretical_h_x100 < u32::MAX);
        }
    }

    #[test]
    fn mean_height_le_n() {
        let v = simulate(20, 100, 42);
        if let TreeVerdict::Ok { mean_height, .. } = v {
            assert!(mean_height <= 20);
        }
    }
}
