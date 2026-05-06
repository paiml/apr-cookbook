//! # Advanced Session-Affinity Router
//!
//! Multi-replica inference: route a session_id to the same replica each
//! time so the KV-cache stays warm. Hash the session_id mod num_replicas.
//! Skip unhealthy replicas (set bit) by stepping forward.
//!
//! Demonstrates the **ADV.34** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Consistent hashing (Karger et al. 1997) for session sticky.
//!
//! Run with: cargo run --example adv_session_affinity_router
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    Replica { index: u32, primary: bool },
    AllUnhealthy,
    InvalidConfig,
}

pub fn route(session_id: &str, healthy_mask: u64, num_replicas: u32) -> RouteVerdict {
    if num_replicas == 0 || num_replicas > 64 {
        return RouteVerdict::InvalidConfig;
    }
    if healthy_mask == 0 {
        return RouteVerdict::AllUnhealthy;
    }
    let h = simple_hash(session_id);
    let primary = h % u64::from(num_replicas);
    if (healthy_mask >> primary) & 1 == 1 {
        return RouteVerdict::Replica {
            index: primary as u32,
            primary: true,
        };
    }
    // Step forward to first healthy replica.
    for offset in 1..num_replicas {
        let idx = (primary + u64::from(offset)) % u64::from(num_replicas);
        if (healthy_mask >> idx) & 1 == 1 {
            return RouteVerdict::Replica {
                index: idx as u32,
                primary: false,
            };
        }
    }
    RouteVerdict::AllUnhealthy
}

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_session_affinity_router")?;

    let healthy = 0b1111u64;
    println!("session_a: {:?}", route("session_a", healthy, 4));
    println!("session_b: {:?}", route("session_b", healthy, 4));
    println!("primary down: {:?}", route("session_a", 0b1110, 4));
    println!("all down: {:?}", route("session_a", 0, 4));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_session_same_replica() {
        let healthy = 0b1111;
        let v1 = route("user42", healthy, 4);
        let v2 = route("user42", healthy, 4);
        assert_eq!(v1, v2);
    }

    #[test]
    fn primary_used_when_healthy() {
        let healthy = 0b1111;
        let v = route("user_a", healthy, 4);
        if let RouteVerdict::Replica { primary, .. } = v {
            assert!(primary);
        }
    }

    #[test]
    fn fallback_when_primary_down() {
        // Force primary down: hash(s)%4 = 2, take healthy_mask = 0b1011 (replica 2 down).
        // Many sessions map to different primaries; just check primary=false comes up.
        let mut found_fallback = false;
        for i in 0..10 {
            let v = route(&format!("u{i}"), 0b1011, 4);
            if let RouteVerdict::Replica { primary: false, .. } = v {
                found_fallback = true;
                break;
            }
        }
        assert!(found_fallback);
    }

    #[test]
    fn all_down_classified() {
        assert_eq!(route("u", 0, 4), RouteVerdict::AllUnhealthy);
    }

    #[test]
    fn zero_replicas_invalid() {
        assert_eq!(route("u", 1, 0), RouteVerdict::InvalidConfig);
    }

    #[test]
    fn over_64_replicas_invalid() {
        assert_eq!(route("u", 1, 65), RouteVerdict::InvalidConfig);
    }

    #[test]
    fn replica_index_in_range() {
        for i in 0..50 {
            let v = route(&format!("s{i}"), 0b1111, 4);
            if let RouteVerdict::Replica { index, .. } = v {
                assert!(index < 4);
            }
        }
    }

    #[test]
    fn empty_session_works() {
        let v = route("", 0b1111, 4);
        assert!(matches!(v, RouteVerdict::Replica { .. }));
    }

    #[test]
    fn single_replica_always_zero() {
        let v = route("any", 0b1, 1);
        if let RouteVerdict::Replica { index, .. } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn deterministic() {
        let a = route("session_42", 0b1111, 4);
        let b = route("session_42", 0b1111, 4);
        assert_eq!(a, b);
    }
}
