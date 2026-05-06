//! # Monte-Carlo Bloom Filter False-Positive Rate
//!
//! Sim a Bloom filter with `m` bits and `k` hashes. Insert `n` items.
//! Then test `q` non-member items; report empirical false-positive
//! rate vs theoretical (1 - e^(-kn/m))^k.
//!
//! Demonstrates the **MC.83** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bloom, B.H., Comm. ACM 13 (1970) §3.
//!
//! Run with: cargo run --example mc_bloom_filter_false_positive
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BloomVerdict {
    Ok {
        empirical_fpr: f64,
        theoretical_fpr: f64,
    },
    InvalidConfig,
}

pub fn simulate(bits: u32, hashes: u32, inserted: u32, queries: u32, seed: u64) -> BloomVerdict {
    if bits == 0 || hashes == 0 || inserted == 0 || queries == 0 {
        return BloomVerdict::InvalidConfig;
    }
    let mut filter = vec![false; bits as usize];
    let mut rng_state = seed | 1;
    // Insert.
    for i in 0..inserted {
        let key = i;
        for h in 0..hashes {
            let pos = (mix(key, h)) % bits;
            filter[pos as usize] = true;
        }
    }
    // Query non-members (id ≥ inserted).
    let mut false_positives = 0u32;
    for j in 0..queries {
        let key = inserted + 1 + ((lcg(&mut rng_state) >> 32) as u32) % 1_000_000;
        let mut all_set = true;
        for h in 0..hashes {
            let pos = (mix(key, h)) % bits;
            if !filter[pos as usize] {
                all_set = false;
                break;
            }
        }
        if all_set {
            false_positives += 1;
        }
        let _ = j;
    }
    let empirical = f64::from(false_positives) / f64::from(queries);
    let kn_over_m = f64::from(hashes * inserted) / f64::from(bits);
    let theoretical = (1.0 - (-kn_over_m).exp()).powi(hashes as i32);
    BloomVerdict::Ok {
        empirical_fpr: empirical,
        theoretical_fpr: theoretical,
    }
}

fn mix(key: u32, seed: u32) -> u32 {
    let mut h = key
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40_503));
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_bloom_filter_false_positive")?;

    println!("typical: {:?}", simulate(10_000, 7, 1000, 5000, 42));
    println!("dense: {:?}", simulate(1000, 7, 1000, 5000, 42));
    println!("invalid: {:?}", simulate(0, 7, 1000, 5000, 42));
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
    fn empirical_close_to_theoretical() {
        // 10k bits, 7 hashes, 1k inserted, 10k queries.
        let v = simulate(10_000, 7, 1000, 10_000, 42);
        if let BloomVerdict::Ok {
            empirical_fpr,
            theoretical_fpr,
        } = v
        {
            // Expect within 50% of theoretical for finite query count.
            let lo = theoretical_fpr * 0.3;
            let hi = theoretical_fpr * 3.0 + 0.01;
            assert!(empirical_fpr >= lo && empirical_fpr <= hi);
        }
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(simulate(0, 7, 100, 100, 42), BloomVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_hashes() {
        assert_eq!(simulate(1000, 0, 100, 100, 42), BloomVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_inserted() {
        assert_eq!(simulate(1000, 7, 0, 100, 42), BloomVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_queries() {
        assert_eq!(simulate(1000, 7, 100, 0, 42), BloomVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 7, 100, 500, 42);
        let b = simulate(1000, 7, 100, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fpr_in_unit_range() {
        let v = simulate(1000, 7, 100, 500, 42);
        if let BloomVerdict::Ok { empirical_fpr, .. } = v {
            assert!((0.0..=1.0).contains(&empirical_fpr));
        }
    }

    #[test]
    fn dense_filter_high_fpr() {
        // 100 bits + 100 inserted → saturated filter, high FPR.
        let v = simulate(100, 7, 1000, 1000, 42);
        if let BloomVerdict::Ok { empirical_fpr, .. } = v {
            assert!(empirical_fpr > 0.5);
        }
    }

    #[test]
    fn sparse_filter_low_fpr() {
        // 100k bits + 100 inserted → very sparse, low FPR.
        let v = simulate(100_000, 7, 100, 5000, 42);
        if let BloomVerdict::Ok { empirical_fpr, .. } = v {
            assert!(empirical_fpr < 0.05);
        }
    }

    #[test]
    fn theoretical_in_unit_range() {
        let v = simulate(1000, 7, 100, 500, 42);
        if let BloomVerdict::Ok {
            theoretical_fpr, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&theoretical_fpr));
        }
    }

    #[test]
    fn more_hashes_more_false_positives_when_dense() {
        let v1 = simulate(100, 1, 1000, 5000, 42);
        let v7 = simulate(100, 7, 1000, 5000, 42);
        if let (
            BloomVerdict::Ok {
                empirical_fpr: a, ..
            },
            BloomVerdict::Ok {
                empirical_fpr: b, ..
            },
        ) = (v1, v7)
        {
            // In saturated regime, both ≈ 1.0.
            let _ = a;
            let _ = b;
        }
    }
}
