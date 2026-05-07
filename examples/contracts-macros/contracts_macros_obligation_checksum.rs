//! # Contracts-Macros Obligation Checksum
//!
//! Compute a deterministic FNV-1a 32-bit checksum over an obligation's
//! id and equation text. Returns the checksum and a "stable" flag if
//! it matches an expected value.
//!
//! Demonstrates the **CMM.142** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Fowler-Noll-Vo hash function (FNV-1a 32-bit, 1991);
//!  RFC ietf-draft-eastlake-fnv-19.
//!
//! Run with: cargo run --example contracts_macros_obligation_checksum
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChecksumVerdict {
    Ok { checksum: u32, stable: bool },
    InvalidConfig,
}

pub fn compute(id: &str, equation: &str, expected: Option<u32>) -> ChecksumVerdict {
    if id.is_empty() || equation.is_empty() {
        return ChecksumVerdict::InvalidConfig;
    }
    let cs = fnv1a_combined(id, equation);
    let stable = match expected {
        Some(v) => v == cs,
        None => true,
    };
    ChecksumVerdict::Ok {
        checksum: cs,
        stable,
    }
}

fn fnv1a_combined(a: &str, b: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in a.bytes().chain(std::iter::once(0)).chain(b.bytes()) {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_checksum")?;

    println!("compute: {:?}", compute("o1", "x + y == z", None));
    println!("invalid: {:?}", compute("", "x", None));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksummer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn deterministic_checksum() {
        let r1 = compute("o", "eq", None);
        let r2 = compute("o", "eq", None);
        assert_eq!(r1, r2);
    }

    #[test]
    fn different_id_different_checksum() {
        if let (ChecksumVerdict::Ok { checksum: a, .. }, ChecksumVerdict::Ok { checksum: b, .. }) =
            (compute("a", "eq", None), compute("b", "eq", None))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn different_equation_different_checksum() {
        if let (ChecksumVerdict::Ok { checksum: a, .. }, ChecksumVerdict::Ok { checksum: b, .. }) =
            (compute("o", "x", None), compute("o", "y", None))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn empty_id_rejected() {
        assert_eq!(compute("", "eq", None), ChecksumVerdict::InvalidConfig);
    }

    #[test]
    fn empty_equation_rejected() {
        assert_eq!(compute("o", "", None), ChecksumVerdict::InvalidConfig);
    }

    #[test]
    fn matching_expected_stable() {
        let r1 = compute("o", "eq", None);
        if let ChecksumVerdict::Ok { checksum, .. } = r1 {
            let r2 = compute("o", "eq", Some(checksum));
            if let ChecksumVerdict::Ok { stable, .. } = r2 {
                assert!(stable);
            }
        }
    }

    #[test]
    fn mismatched_expected_unstable() {
        let r = compute("o", "eq", Some(0));
        if let ChecksumVerdict::Ok { stable, .. } = r {
            assert!(!stable);
        }
    }

    #[test]
    fn no_expected_defaults_stable() {
        let r = compute("o", "eq", None);
        if let ChecksumVerdict::Ok { stable, .. } = r {
            assert!(stable);
        }
    }

    #[test]
    fn delimiter_isolates_id_eq() {
        // Without the null-byte delimiter, ("ab","c") and ("a","bc")
        // would collide. With it, they should differ.
        if let (ChecksumVerdict::Ok { checksum: a, .. }, ChecksumVerdict::Ok { checksum: b, .. }) =
            (compute("ab", "c", None), compute("a", "bc", None))
        {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn unicode_supported() {
        let r = compute("café", "résumé", None);
        assert!(matches!(r, ChecksumVerdict::Ok { .. }));
    }

    #[test]
    fn fnv_offset_used() {
        // Single-char id produces non-zero hash.
        let r = compute("a", "b", None);
        if let ChecksumVerdict::Ok { checksum, .. } = r {
            assert_ne!(checksum, 0);
        }
    }
}
