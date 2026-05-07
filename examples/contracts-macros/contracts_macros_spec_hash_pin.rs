//! # Contracts-Macros Spec Hash Pin
//!
//! Compute a spec content hash (FNV-1a 64-bit over equation list)
//! and compare to a pinned hash. Returns `Match` or `Drift` verdict
//! with the observed hash.
//!
//! Demonstrates the **CMM.149** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cargo.lock content addressing; pip hash-pin (PEP 540)
//!  for reproducibility.
//!
//! Run with: cargo run --example contracts_macros_spec_hash_pin
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HashPinVerdict {
    Match { observed_hash: u64 },
    Drift { observed_hash: u64, pinned: u64 },
    InvalidConfig,
}

pub fn check(equations: &[&str], pinned: u64) -> HashPinVerdict {
    if equations.is_empty() {
        return HashPinVerdict::InvalidConfig;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for eq in equations {
        for byte in eq.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        // Inter-equation delimiter
        hash ^= 0;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if hash == pinned {
        HashPinVerdict::Match {
            observed_hash: hash,
        }
    } else {
        HashPinVerdict::Drift {
            observed_hash: hash,
            pinned,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_spec_hash_pin")?;

    let v = check(&["x + 1 == y"], 0);
    println!("drift: {v:?}");
    if let HashPinVerdict::Drift { observed_hash, .. } = v {
        println!("match: {:?}", check(&["x + 1 == y"], observed_hash));
    }
    println!("invalid: {:?}", check(&[], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn deterministic_hash() {
        let r1 = check(&["x"], 0);
        let r2 = check(&["x"], 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn different_equations_different_hash() {
        if let (
            HashPinVerdict::Drift {
                observed_hash: a, ..
            },
            HashPinVerdict::Drift {
                observed_hash: b, ..
            },
        ) = (check(&["x"], 0), check(&["y"], 0))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn matching_pin_returns_match() {
        if let HashPinVerdict::Drift { observed_hash, .. } = check(&["x"], 0) {
            let v = check(&["x"], observed_hash);
            assert!(matches!(v, HashPinVerdict::Match { .. }));
        }
    }

    #[test]
    fn empty_equations_rejected() {
        assert_eq!(check(&[], 0), HashPinVerdict::InvalidConfig);
    }

    #[test]
    fn order_matters() {
        if let (
            HashPinVerdict::Drift {
                observed_hash: a, ..
            },
            HashPinVerdict::Drift {
                observed_hash: b, ..
            },
        ) = (check(&["x", "y"], 0), check(&["y", "x"], 0))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn many_equations_handled() {
        let eqs: Vec<&str> = (0..30).map(|_| "eq").collect();
        let v = check(&eqs, 0);
        assert!(matches!(v, HashPinVerdict::Drift { .. }));
    }

    #[test]
    fn unicode_equation_supported() {
        let v = check(&["café = π"], 0);
        assert!(matches!(v, HashPinVerdict::Drift { .. }));
    }

    #[test]
    fn drift_returns_both_hashes() {
        let v = check(&["x"], 12345);
        if let HashPinVerdict::Drift { pinned, .. } = v {
            assert_eq!(pinned, 12345);
        }
    }

    #[test]
    fn fnv_offset_used() {
        // Single equation produces non-zero hash.
        if let HashPinVerdict::Drift { observed_hash, .. } = check(&["a"], 999) {
            assert_ne!(observed_hash, 0);
        }
    }

    #[test]
    fn delimiter_separates_eqs() {
        // Without inter-equation delimiter, ("ab","c") and ("a","bc") would collide.
        if let (
            HashPinVerdict::Drift {
                observed_hash: a, ..
            },
            HashPinVerdict::Drift {
                observed_hash: b, ..
            },
        ) = (check(&["ab", "c"], 0), check(&["a", "bc"], 0))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn match_observed_equals_pinned() {
        if let HashPinVerdict::Drift { observed_hash, .. } = check(&["x"], 0) {
            if let HashPinVerdict::Match { observed_hash: m } = check(&["x"], observed_hash) {
                assert_eq!(m, observed_hash);
            }
        }
    }
}
