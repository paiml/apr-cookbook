//! # Monte-Carlo Kademlia DHT Lookup Hops
//!
//! Sim Kademlia DHT routing: each node knows k buckets of peers
//! organized by XOR distance. Reports avg lookup hops needed.
//!
//! Demonstrates the **MC.128** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Maymounkov & Mazières, "Kademlia: A P2P Information
//!  System Based on the XOR Metric" (IPTPS 2002).
//!
//! Run with: cargo run --example mc_kademlia_routing_lookup
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KademliaVerdict {
    Ok { avg_hops: f64, max_hops: u32 },
    InvalidConfig,
}

pub fn simulate(lookups: u32, network_size_bits: u32, seed: u64) -> KademliaVerdict {
    if lookups == 0 || network_size_bits == 0 || network_size_bits > 32 {
        return KademliaVerdict::InvalidConfig;
    }
    // Theoretical: O(log_2(N)) hops per lookup; we sample.
    let mut total_hops: u64 = 0;
    let mut max_hops: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..lookups {
        let target = (lcg(&mut rng_state) >> 32) as u32;
        let start = (lcg(&mut rng_state) >> 32) as u32;
        let xor_distance = target ^ start;
        // Hops = number of significant bits in xor distance.
        let hops = if xor_distance == 0 {
            0
        } else {
            // 32-bit XOR has up to 32 significant bits.
            let bits = 32u32.saturating_sub(xor_distance.leading_zeros());
            bits.min(network_size_bits)
        };
        total_hops += u64::from(hops);
        if hops > max_hops {
            max_hops = hops;
        }
    }
    KademliaVerdict::Ok {
        avg_hops: total_hops as f64 / f64::from(lookups),
        max_hops,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kademlia_routing_lookup")?;

    println!("16-bit space: {:?}", simulate(1000, 16, 42));
    println!("32-bit space: {:?}", simulate(1000, 32, 42));
    println!("invalid: {:?}", simulate(0, 16, 42));
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
    fn larger_space_more_hops() {
        let small = simulate(1000, 8, 42);
        let big = simulate(1000, 32, 42);
        if let (KademliaVerdict::Ok { avg_hops: s, .. }, KademliaVerdict::Ok { avg_hops: b, .. }) =
            (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn invalid_zero_lookups() {
        assert_eq!(simulate(0, 16, 42), KademliaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(simulate(100, 0, 42), KademliaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_bits_above_32() {
        assert_eq!(simulate(100, 64, 42), KademliaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 16, 42);
        let b = simulate(500, 16, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_hops_le_network_bits() {
        let v = simulate(500, 16, 42);
        if let KademliaVerdict::Ok { max_hops, .. } = v {
            assert!(max_hops <= 16);
        }
    }

    #[test]
    fn avg_hops_le_max() {
        let v = simulate(500, 16, 42);
        if let KademliaVerdict::Ok {
            avg_hops, max_hops, ..
        } = v
        {
            assert!(avg_hops <= f64::from(max_hops));
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(500, 16, 42);
        if let KademliaVerdict::Ok { avg_hops, .. } = v {
            assert!(avg_hops.is_finite());
        }
    }

    #[test]
    fn avg_hops_nonneg() {
        let v = simulate(500, 16, 42);
        if let KademliaVerdict::Ok { avg_hops, .. } = v {
            assert!(avg_hops >= 0.0);
        }
    }

    #[test]
    fn small_lookup_works() {
        let v = simulate(1, 16, 42);
        assert!(matches!(v, KademliaVerdict::Ok { .. }));
    }

    #[test]
    fn many_lookups_handled() {
        let v = simulate(10_000, 32, 42);
        assert!(matches!(v, KademliaVerdict::Ok { .. }));
    }

    #[test]
    fn boundary_32_bits_works() {
        let v = simulate(100, 32, 42);
        assert!(matches!(v, KademliaVerdict::Ok { .. }));
    }
}
