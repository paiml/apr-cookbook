//! # Monte-Carlo LRU + Admission Filter
//!
//! Compare LRU cache hit-rate vs LRU + admission filter (TinyLFU-style:
//! admit only items seen >= `admit_threshold` times in last
//! `window_size` queries).
//!
//! Demonstrates the **MC.103** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TinyLFU (Einziger, Friedman, Manes, ACM TOS 2017);
//!  W-TinyLFU admission policy.
//!
//! Run with: cargo run --example mc_lru_admission_filter
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok {
        lru_hit_rate: f64,
        filtered_hit_rate: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    queries: u32,
    keyspace: u32,
    cache_size: u32,
    admit_threshold: u32,
    window_size: u32,
    seed: u64,
) -> FilterVerdict {
    if queries == 0 || keyspace == 0 || cache_size == 0 || admit_threshold == 0 || window_size == 0
    {
        return FilterVerdict::InvalidConfig;
    }
    let mut lru: VecDeque<u32> = VecDeque::with_capacity(cache_size as usize);
    let mut filt: VecDeque<u32> = VecDeque::with_capacity(cache_size as usize);
    let mut window: VecDeque<u32> = VecDeque::with_capacity(window_size as usize);
    let mut window_counts: BTreeMap<u32, u32> = BTreeMap::new();
    let mut lru_hits = 0u32;
    let mut filt_hits = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..queries {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % keyspace;
        // Update window.
        if window.len() as u32 >= window_size {
            if let Some(old) = window.pop_front() {
                let c = window_counts.entry(old).or_insert(0);
                *c -= 1;
                if *c == 0 {
                    window_counts.remove(&old);
                }
            }
        }
        window.push_back(key);
        *window_counts.entry(key).or_insert(0) += 1;
        // LRU access.
        if let Some(pos) = lru.iter().position(|k| *k == key) {
            lru.remove(pos);
            lru.push_back(key);
            lru_hits += 1;
        } else {
            if lru.len() >= cache_size as usize {
                lru.pop_front();
            }
            lru.push_back(key);
        }
        // Filtered LRU access.
        if let Some(pos) = filt.iter().position(|k| *k == key) {
            filt.remove(pos);
            filt.push_back(key);
            filt_hits += 1;
        } else if window_counts.get(&key).copied().unwrap_or(0) >= admit_threshold {
            if filt.len() >= cache_size as usize {
                filt.pop_front();
            }
            filt.push_back(key);
        }
    }
    FilterVerdict::Ok {
        lru_hit_rate: f64::from(lru_hits) / f64::from(queries),
        filtered_hit_rate: f64::from(filt_hits) / f64::from(queries),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_lru_admission_filter")?;

    println!("small set: {:?}", simulate(10_000, 100, 20, 2, 100, 42));
    println!("wide set: {:?}", simulate(10_000, 10_000, 20, 2, 100, 42));
    println!("invalid: {:?}", simulate(0, 100, 20, 2, 100, 42));
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
    fn invalid_zero_queries() {
        assert_eq!(
            simulate(0, 100, 20, 2, 100, 42),
            FilterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(
            simulate(100, 100, 0, 2, 100, 42),
            FilterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_threshold() {
        assert_eq!(
            simulate(100, 100, 20, 0, 100, 42),
            FilterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(
            simulate(100, 100, 20, 2, 0, 42),
            FilterVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 100, 20, 2, 100, 42);
        let b = simulate(500, 100, 20, 2, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(500, 100, 20, 2, 100, 42);
        if let FilterVerdict::Ok {
            lru_hit_rate,
            filtered_hit_rate,
        } = v
        {
            assert!((0.0..=1.0).contains(&lru_hit_rate));
            assert!((0.0..=1.0).contains(&filtered_hit_rate));
        }
    }

    #[test]
    fn cache_at_keyspace_size_full_lru_hits() {
        let v = simulate(10_000, 5, 5, 1, 10, 42);
        if let FilterVerdict::Ok { lru_hit_rate, .. } = v {
            assert!(lru_hit_rate > 0.99);
        }
    }

    #[test]
    fn filter_at_threshold_one_admits_all() {
        // threshold=1 means all keys admitted on first sighting → like LRU.
        let v = simulate(2000, 100, 20, 1, 100, 42);
        if let FilterVerdict::Ok {
            lru_hit_rate,
            filtered_hit_rate,
        } = v
        {
            assert!((lru_hit_rate - filtered_hit_rate).abs() < 0.05);
        }
    }

    #[test]
    fn higher_threshold_filters_more() {
        let v_lo = simulate(2000, 100, 20, 1, 100, 42);
        let v_hi = simulate(2000, 100, 20, 5, 100, 42);
        if let (
            FilterVerdict::Ok {
                filtered_hit_rate: lo,
                ..
            },
            FilterVerdict::Ok {
                filtered_hit_rate: hi,
                ..
            },
        ) = (v_lo, v_hi)
        {
            // Higher threshold → admits fewer keys → likely lower hit rate.
            assert!(lo >= hi - 0.10);
        }
    }

    #[test]
    fn first_query_no_hits() {
        let v = simulate(1, 100, 20, 2, 100, 42);
        if let FilterVerdict::Ok {
            lru_hit_rate,
            filtered_hit_rate,
        } = v
        {
            assert_eq!(lru_hit_rate, 0.0);
            assert_eq!(filtered_hit_rate, 0.0);
        }
    }

    #[test]
    fn small_keyspace_high_hit_rates() {
        let v = simulate(10_000, 10, 5, 2, 50, 42);
        if let FilterVerdict::Ok { lru_hit_rate, .. } = v {
            assert!(lru_hit_rate > 0.30);
        }
    }
}
