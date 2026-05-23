//! # Monte-Carlo Erdős-Rényi Connectedness
//!
//! Sim Erdős-Rényi G(n, p) random graphs. Reports fraction of trials
//! producing connected graphs (sharp transition near p = ln(n)/n).
//!
//! Demonstrates the **MC.93** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Erdős, P. & Rényi, A. (1959) Publ. Math. Debrecen 6;
//!  giant component theorem (Bollobás 1981).
//!
//! Run with: cargo run --example mc_random_graph_connectedness
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GraphVerdict {
    Ok {
        connected_rate: f64,
        avg_components: f64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, nodes: u32, edge_prob: f64, seed: u64) -> GraphVerdict {
    if trials == 0 || nodes < 2 || !(0.0..=1.0).contains(&edge_prob) {
        return GraphVerdict::InvalidConfig;
    }
    let n = nodes as usize;
    let mut connected_count = 0u32;
    let mut total_components: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        // Build adjacency (upper triangle).
        let mut parent: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
                if r < edge_prob {
                    union(&mut parent, i, j);
                }
            }
        }
        // Count components.
        let mut roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
        roots.sort_unstable();
        roots.dedup();
        let components = roots.len();
        total_components += components as u64;
        if components == 1 {
            connected_count += 1;
        }
    }
    GraphVerdict::Ok {
        connected_rate: f64::from(connected_count) / f64::from(trials),
        avg_components: total_components as f64 / f64::from(trials),
    }
}

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], x: usize, y: usize) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx != ry {
        parent[rx] = ry;
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_graph_connectedness")?;

    println!("dense p=0.5: {:?}", simulate(50, 20, 0.5, 42));
    println!("sparse p=0.05: {:?}", simulate(50, 20, 0.05, 42));
    println!("invalid: {:?}", simulate(0, 20, 0.5, 42));
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
    fn dense_graph_mostly_connected() {
        let v = simulate(100, 20, 0.5, 42);
        if let GraphVerdict::Ok { connected_rate, .. } = v {
            assert!(connected_rate > 0.95);
        }
    }

    #[test]
    fn sparse_graph_rarely_connected() {
        let v = simulate(100, 20, 0.01, 42);
        if let GraphVerdict::Ok { connected_rate, .. } = v {
            assert!(connected_rate < 0.10);
        }
    }

    #[test]
    fn empty_graph_n_components() {
        let v = simulate(1, 5, 0.0, 42);
        if let GraphVerdict::Ok { avg_components, .. } = v {
            assert_eq!(avg_components, 5.0);
        }
    }

    #[test]
    fn complete_graph_one_component() {
        let v = simulate(1, 5, 1.0, 42);
        if let GraphVerdict::Ok {
            avg_components,
            connected_rate,
        } = v
        {
            assert_eq!(avg_components, 1.0);
            assert_eq!(connected_rate, 1.0);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 5, 0.5, 42), GraphVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_nodes() {
        assert_eq!(simulate(10, 1, 0.5, 42), GraphVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(10, 5, 1.5, 42), GraphVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 10, 0.3, 42);
        let b = simulate(20, 10, 0.3, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(20, 10, 0.3, 42);
        if let GraphVerdict::Ok { connected_rate, .. } = v {
            assert!((0.0..=1.0).contains(&connected_rate));
        }
    }

    #[test]
    fn higher_p_more_connected() {
        let lo = simulate(50, 10, 0.05, 42);
        let hi = simulate(50, 10, 0.5, 42);
        if let (
            GraphVerdict::Ok {
                connected_rate: l, ..
            },
            GraphVerdict::Ok {
                connected_rate: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn avg_components_at_least_one() {
        let v = simulate(20, 10, 0.5, 42);
        if let GraphVerdict::Ok { avg_components, .. } = v {
            assert!(avg_components >= 1.0);
        }
    }
}
