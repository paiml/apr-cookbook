//! # Monte-Carlo Reservoir Sampling (Algorithm R)
//!
//! Sample k items uniformly at random from a stream of unknown length
//! using Vitter's algorithm R. Returns the final reservoir contents
//! sorted ascending.
//!
//! Demonstrates the **MC.136** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Vitter, "Random Sampling with a Reservoir" ACM TOMS
//!  11(1):37-57 (1985); Knuth TAOCP §3.4.2 Algorithm S/R.
//!
//! Run with: cargo run --example mc_reservoir_sample
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReservoirVerdict {
    Ok {
        reservoir: Vec<u32>,
        items_seen: u32,
    },
    InvalidConfig,
}

pub fn sample(stream: &[u32], k: u32, seed: u64) -> ReservoirVerdict {
    if k == 0 || stream.is_empty() {
        return ReservoirVerdict::InvalidConfig;
    }
    let k_us = k as usize;
    let mut state = seed | 1;
    let mut reservoir: Vec<u32> = Vec::with_capacity(k_us);
    for (i, item) in stream.iter().enumerate() {
        if i < k_us {
            reservoir.push(*item);
        } else {
            let j = (lcg(&mut state) as usize) % (i + 1);
            if j < k_us {
                reservoir[j] = *item;
            }
        }
    }
    reservoir.sort_unstable();
    ReservoirVerdict::Ok {
        reservoir,
        items_seen: stream.len() as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_reservoir_sample")?;

    let stream: Vec<u32> = (1..=100).collect();
    println!("k=5: {:?}", sample(&stream, 5, 42));
    println!("invalid: {:?}", sample(&[], 5, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn reservoir_size_equals_k_when_stream_long_enough() {
        let stream: Vec<u32> = (1..=100).collect();
        let v = sample(&stream, 5, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            assert_eq!(reservoir.len(), 5);
        }
    }

    #[test]
    fn reservoir_size_equals_stream_when_short() {
        let stream = vec![1u32, 2, 3];
        let v = sample(&stream, 10, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            assert_eq!(reservoir.len(), 3);
        }
    }

    #[test]
    fn invalid_zero_k() {
        assert_eq!(sample(&[1, 2, 3], 0, 42), ReservoirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_stream() {
        assert_eq!(sample(&[], 5, 42), ReservoirVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let stream: Vec<u32> = (1..=50).collect();
        let a = sample(&stream, 5, 42);
        let b = sample(&stream, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn reservoir_items_drawn_from_stream() {
        let stream: Vec<u32> = (1..=20).collect();
        let v = sample(&stream, 5, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            for r in &reservoir {
                assert!(stream.contains(r));
            }
        }
    }

    #[test]
    fn items_seen_equals_stream_length() {
        let stream: Vec<u32> = (1..=50).collect();
        let v = sample(&stream, 5, 42);
        if let ReservoirVerdict::Ok { items_seen, .. } = v {
            assert_eq!(items_seen, 50);
        }
    }

    #[test]
    fn reservoir_sorted() {
        let stream: Vec<u32> = (1..=20).collect();
        let v = sample(&stream, 10, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            for w in reservoir.windows(2) {
                assert!(w[0] <= w[1]);
            }
        }
    }

    #[test]
    fn k_equals_stream_returns_all() {
        let stream = vec![3u32, 1, 2];
        let v = sample(&stream, 3, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            assert_eq!(reservoir, vec![1, 2, 3]);
        }
    }

    #[test]
    fn different_seed_different_sample() {
        let stream: Vec<u32> = (1..=100).collect();
        let a = sample(&stream, 5, 42);
        let b = sample(&stream, 5, 99);
        // Not strictly required but extremely likely.
        assert!(a != b);
    }

    #[test]
    fn k_one_handled() {
        let stream: Vec<u32> = (1..=10).collect();
        let v = sample(&stream, 1, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            assert_eq!(reservoir.len(), 1);
        }
    }

    #[test]
    fn many_items_handled() {
        let stream: Vec<u32> = (1..=10_000).collect();
        let v = sample(&stream, 10, 42);
        if let ReservoirVerdict::Ok { reservoir, .. } = v {
            assert_eq!(reservoir.len(), 10);
        }
    }
}
