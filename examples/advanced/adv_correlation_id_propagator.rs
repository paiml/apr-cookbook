//! # Advanced Correlation-ID Propagator
//!
//! Propagate `X-Correlation-ID` through downstream calls. If the
//! request lacks one, generate a fresh ID. Reject malformed IDs to
//! prevent log poisoning.
//!
//! Demonstrates the **ADV.42** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: W3C Trace Context (traceparent header).
//!
//! Run with: cargo run --example adv_correlation_id_propagator
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PropagateVerdict {
    Reuse { id: String },
    Generated { id: String },
    Malformed { reason: &'static str },
}

pub fn propagate(incoming_id: Option<&str>, fallback_seed: &str) -> PropagateVerdict {
    if let Some(id) = incoming_id {
        return match validate(id) {
            Some(reason) => PropagateVerdict::Malformed { reason },
            None => PropagateVerdict::Reuse { id: id.to_string() },
        };
    }
    PropagateVerdict::Generated {
        id: format!("gen-{}", short_hash(fallback_seed)),
    }
}

fn validate(id: &str) -> Option<&'static str> {
    if id.is_empty() {
        return Some("empty");
    }
    if id.len() > 64 {
        return Some("too long");
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Some("invalid characters");
    }
    None
}

fn short_hash(seed: &str) -> String {
    let mut h: u64 = 14695981039346656037;
    for b in seed.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(1099511628211);
    }
    format!("{h:016x}")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_correlation_id_propagator")?;

    println!("reuse: {:?}", propagate(Some("abc-123"), "seed"));
    println!("generate: {:?}", propagate(None, "seed_request_5"));
    println!("malformed: {:?}", propagate(Some(""), "seed"));
    println!(
        "malformed too long: {:?}",
        propagate(Some(&"x".repeat(100)), "seed")
    );
    println!("malformed chars: {:?}", propagate(Some("bad id"), "seed"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_id_reused() {
        let v = propagate(Some("abc-123"), "seed");
        if let PropagateVerdict::Reuse { id } = v {
            assert_eq!(id, "abc-123");
        }
    }

    #[test]
    fn no_id_generates() {
        let v = propagate(None, "seed");
        assert!(matches!(v, PropagateVerdict::Generated { .. }));
    }

    #[test]
    fn empty_id_malformed() {
        let v = propagate(Some(""), "seed");
        assert!(matches!(v, PropagateVerdict::Malformed { .. }));
    }

    #[test]
    fn too_long_malformed() {
        let v = propagate(Some(&"x".repeat(100)), "seed");
        assert!(matches!(v, PropagateVerdict::Malformed { .. }));
    }

    #[test]
    fn special_chars_malformed() {
        let v = propagate(Some("bad id"), "seed");
        assert!(matches!(v, PropagateVerdict::Malformed { .. }));
    }

    #[test]
    fn underscore_allowed() {
        let v = propagate(Some("abc_123"), "seed");
        assert!(matches!(v, PropagateVerdict::Reuse { .. }));
    }

    #[test]
    fn boundary_at_64_chars_ok() {
        let id = "x".repeat(64);
        let v = propagate(Some(&id), "seed");
        assert!(matches!(v, PropagateVerdict::Reuse { .. }));
    }

    #[test]
    fn just_over_64_chars_malformed() {
        let id = "x".repeat(65);
        let v = propagate(Some(&id), "seed");
        assert!(matches!(v, PropagateVerdict::Malformed { .. }));
    }

    #[test]
    fn generated_id_has_prefix() {
        let v = propagate(None, "seed_request");
        if let PropagateVerdict::Generated { id } = v {
            assert!(id.starts_with("gen-"));
        }
    }

    #[test]
    fn different_seeds_different_ids() {
        let a = propagate(None, "seed_a");
        let b = propagate(None, "seed_b");
        if let (PropagateVerdict::Generated { id: ia }, PropagateVerdict::Generated { id: ib }) =
            (a, b)
        {
            assert_ne!(ia, ib);
        }
    }

    #[test]
    fn deterministic() {
        let a = propagate(Some("abc"), "seed");
        let b = propagate(Some("abc"), "seed");
        assert_eq!(a, b);
    }
}
