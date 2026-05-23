//! # Monte-Carlo Preferential Attachment Network Growth
//!
//! Sim Barabási-Albert preferential-attachment network growth: each
//! new node connects to m existing nodes with probability proportional
//! to their degree. Returns max-degree (hub) and degree-distribution
//! tail count.
//!
//! Demonstrates the **MC.184** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Barabási & Albert, "Emergence of scaling in random
//!  networks" Science 286 (1999); scale-free network theory.
//!
//! Run with: cargo run --example mc_preferential_attachment_growth
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BaVerdict {
    Ok {
        max_degree: u32,
        high_degree_count: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_nodes: u32, m_edges: u32, seed: u64) -> BaVerdict {
    if n_nodes < 10 || m_edges < 1 || m_edges >= n_nodes {
        return BaVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut degrees: Vec<u32> = vec![0; n_nodes as usize];
    // Initial m+1 nodes form a complete graph.
    let init = (m_edges + 1) as usize;
    for d in degrees.iter_mut().take(init) {
        *d = m_edges;
    }
    // Each new node attaches to m existing nodes proportionally to degree.
    for new_node in init..n_nodes as usize {
        let total_degree: u32 = degrees[..new_node].iter().sum();
        for _ in 0..m_edges {
            let r = lcg(&mut state) % total_degree.max(1) as u64;
            let mut cum = 0u32;
            let mut chosen = 0usize;
            for (i, d) in degrees[..new_node].iter().enumerate() {
                cum += *d;
                if r < cum as u64 {
                    chosen = i;
                    break;
                }
            }
            degrees[chosen] += 1;
            degrees[new_node] += 1;
        }
    }
    let max = *degrees.iter().max().unwrap_or(&0);
    let threshold = max / 2;
    let high_count = degrees.iter().filter(|d| **d > threshold).count() as u32;
    BaVerdict::Ok {
        max_degree: max,
        high_degree_count: high_count,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_preferential_attachment_growth")?;

    println!("n=100, m=2: {:?}", simulate(100, 2, 42));
    println!("invalid: {:?}", simulate(5, 2, 42));
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
    fn invalid_too_small_network() {
        assert_eq!(simulate(5, 2, 42), BaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_edges() {
        assert_eq!(simulate(50, 0, 42), BaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_m_ge_n() {
        assert_eq!(simulate(10, 10, 42), BaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 2, 42);
        let b = simulate(50, 2, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_degree_at_least_m() {
        let v = simulate(100, 3, 42);
        if let BaVerdict::Ok { max_degree, .. } = v {
            assert!(max_degree >= 3);
        }
    }

    #[test]
    fn larger_network_more_concentration() {
        // BA networks: hub-dominance grows with N.
        let small = simulate(30, 2, 42);
        let large = simulate(200, 2, 42);
        if let (BaVerdict::Ok { max_degree: s, .. }, BaVerdict::Ok { max_degree: l, .. }) =
            (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn high_degree_count_at_least_one() {
        let v = simulate(100, 2, 42);
        if let BaVerdict::Ok {
            high_degree_count, ..
        } = v
        {
            assert!(high_degree_count >= 1);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 1, 42);
        assert!(matches!(v, BaVerdict::Ok { .. }));
    }

    #[test]
    fn many_nodes_handled() {
        let v = simulate(500, 3, 42);
        assert!(matches!(v, BaVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 2, 42);
        let b = simulate(100, 2, 999);
        assert!(a != b);
    }

    #[test]
    fn max_degree_bounded_by_total_edges() {
        // Total edges added = m * (n - m - 1) + initial edges.
        // Max single node degree can be at most this.
        let v = simulate(50, 2, 42);
        if let BaVerdict::Ok { max_degree, .. } = v {
            // Max possible degree is bounded by n.
            assert!(max_degree <= 100);
        }
    }
}
