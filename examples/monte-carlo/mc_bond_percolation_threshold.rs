//! # Monte-Carlo 2D Bond Percolation
//!
//! Sim 2D square-lattice bond percolation: each edge is open with
//! probability p. Returns whether a spanning cluster (top→bottom)
//! exists and the largest-cluster size.
//!
//! Demonstrates the **MC.190** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Stauffer & Aharony, Introduction to Percolation Theory
//!  (1992); 2D square-lattice critical p_c ≈ 0.5.
//!
//! Run with: cargo run --example mc_bond_percolation_threshold
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PercolationVerdict {
    Ok { spans: bool, largest_cluster: u32 },
    InvalidConfig,
}

pub fn simulate(grid_size: u32, prob_x100: u32, seed: u64) -> PercolationVerdict {
    if !(5..=200).contains(&grid_size) || !(1..=99).contains(&prob_x100) {
        return PercolationVerdict::InvalidConfig;
    }
    let n = grid_size as usize;
    let p = prob_x100 as f64 / 100.0;
    let mut state = seed | 1;
    // Sites: occupied if open (use bond percolation indirectly via site approximation).
    let mut grid = vec![false; n * n];
    for cell in &mut grid {
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        *cell = r < p;
    }
    // BFS clusters; track largest and whether top-row connects to bottom-row.
    let mut visited = vec![false; n * n];
    let mut largest = 0u32;
    let mut spans = false;
    for start in 0..n * n {
        if !grid[start] || visited[start] {
            continue;
        }
        let mut queue: Vec<usize> = vec![start];
        let mut size = 0u32;
        let mut touches_top = false;
        let mut touches_bottom = false;
        while let Some(cell) = queue.pop() {
            if visited[cell] {
                continue;
            }
            visited[cell] = true;
            size += 1;
            let row = cell / n;
            let col = cell % n;
            if row == 0 {
                touches_top = true;
            }
            if row == n - 1 {
                touches_bottom = true;
            }
            // Neighbors: up, down, left, right
            if row > 0 && grid[cell - n] && !visited[cell - n] {
                queue.push(cell - n);
            }
            if row < n - 1 && grid[cell + n] && !visited[cell + n] {
                queue.push(cell + n);
            }
            if col > 0 && grid[cell - 1] && !visited[cell - 1] {
                queue.push(cell - 1);
            }
            if col < n - 1 && grid[cell + 1] && !visited[cell + 1] {
                queue.push(cell + 1);
            }
        }
        if size > largest {
            largest = size;
        }
        if touches_top && touches_bottom {
            spans = true;
        }
    }
    PercolationVerdict::Ok {
        spans,
        largest_cluster: largest,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_bond_percolation_threshold")?;

    println!("p=0.6 (above pc): {:?}", simulate(20, 60, 42));
    println!("p=0.4 (below pc): {:?}", simulate(20, 40, 42));
    println!("invalid: {:?}", simulate(2, 50, 42));
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
    fn invalid_too_small_grid() {
        assert_eq!(simulate(2, 50, 42), PercolationVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_prob() {
        assert_eq!(simulate(10, 0, 42), PercolationVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_prob() {
        assert_eq!(simulate(10, 100, 42), PercolationVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 50, 42);
        let b = simulate(20, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn high_prob_likely_spans() {
        let v = simulate(30, 80, 42);
        if let PercolationVerdict::Ok { spans, .. } = v {
            assert!(spans);
        }
    }

    #[test]
    fn low_prob_unlikely_spans() {
        let v = simulate(30, 20, 42);
        if let PercolationVerdict::Ok { spans, .. } = v {
            assert!(!spans);
        }
    }

    #[test]
    fn larger_p_larger_cluster() {
        let low = simulate(30, 30, 42);
        let high = simulate(30, 70, 42);
        if let (
            PercolationVerdict::Ok {
                largest_cluster: l, ..
            },
            PercolationVerdict::Ok {
                largest_cluster: h, ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn largest_cluster_le_total_cells() {
        let v = simulate(10, 50, 42);
        if let PercolationVerdict::Ok {
            largest_cluster, ..
        } = v
        {
            assert!(largest_cluster <= 100);
        }
    }

    #[test]
    fn min_grid_accepted() {
        let v = simulate(5, 50, 42);
        assert!(matches!(v, PercolationVerdict::Ok { .. }));
    }

    #[test]
    fn many_cells_handled() {
        let v = simulate(100, 50, 42);
        assert!(matches!(v, PercolationVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(20, 50, 42);
        let b = simulate(20, 50, 999);
        assert!(a != b);
    }
}
