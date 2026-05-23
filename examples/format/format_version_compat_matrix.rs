//! # Format APR Reader/Bundle Version Compat Matrix
//!
//! Reader major/minor → bundle major/minor support matrix:
//! - reader == bundle: full support
//! - reader.major > bundle.major within deprecation window: read-only
//! - reader.major < bundle.major: REJECT
//! - reader >= bundle (within same major) but minor newer: forward-compat
//!
//! Plus deprecation-window check: if reader.major - bundle.major > 1,
//! suggest upgrade to current bundle format.
//!
//! Demonstrates the **FMT.25** recipe for PMAT-136 (format round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: APR format-version policy.
//!
//! Run with: cargo run --example format_version_compat_matrix
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompatLevel {
    FullReadWrite,
    ReadOnly { reason: &'static str },
    Reject { reason: &'static str },
    UpgradeRequired { from_major: u32, to_major: u32 },
}

const DEPRECATION_GRACE_MAJORS: u32 = 1;

pub fn matrix(
    reader_major: u32,
    reader_minor: u32,
    bundle_major: u32,
    bundle_minor: u32,
) -> CompatLevel {
    if reader_major < bundle_major {
        return CompatLevel::Reject {
            reason: "reader is older major than bundle",
        };
    }
    let major_gap = reader_major - bundle_major;
    if major_gap > DEPRECATION_GRACE_MAJORS {
        return CompatLevel::UpgradeRequired {
            from_major: bundle_major,
            to_major: reader_major,
        };
    }
    if major_gap == DEPRECATION_GRACE_MAJORS {
        return CompatLevel::ReadOnly {
            reason: "deprecated bundle format — read-only",
        };
    }
    if reader_minor < bundle_minor {
        return CompatLevel::ReadOnly {
            reason: "reader minor older — unknown fields silently ignored",
        };
    }
    CompatLevel::FullReadWrite
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_version_compat_matrix")?;

    let cases = [
        (2u32, 0u32, 2u32, 0u32),
        (3, 1, 2, 0),
        (4, 0, 2, 0),
        (1, 0, 2, 0),
        (2, 0, 2, 1),
    ];
    for (rma, rmi, bma, bmi) in cases {
        println!(
            "reader {rma}.{rmi} reads bundle {bma}.{bmi}: {:?}",
            matrix(rma, rmi, bma, bmi)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_version_full_rw() {
        assert_eq!(matrix(2, 0, 2, 0), CompatLevel::FullReadWrite);
    }

    #[test]
    fn newer_minor_reader_full_rw() {
        // Reader 2.1 reads bundle 2.0 → all known fields.
        assert_eq!(matrix(2, 1, 2, 0), CompatLevel::FullReadWrite);
    }

    #[test]
    fn older_minor_reader_read_only() {
        let v = matrix(2, 0, 2, 1);
        assert!(matches!(v, CompatLevel::ReadOnly { .. }));
    }

    #[test]
    fn one_major_newer_reader_read_only() {
        // Reader 3.x reading bundle 2.x → deprecated, read-only.
        let v = matrix(3, 0, 2, 0);
        assert!(matches!(v, CompatLevel::ReadOnly { .. }));
    }

    #[test]
    fn two_majors_newer_upgrade_required() {
        // Reader 4.x reading bundle 2.x → out of grace, upgrade.
        let v = matrix(4, 0, 2, 0);
        assert!(matches!(v, CompatLevel::UpgradeRequired { .. }));
    }

    #[test]
    fn older_major_reader_rejected() {
        let v = matrix(1, 0, 2, 0);
        assert!(matches!(v, CompatLevel::Reject { .. }));
    }

    #[test]
    fn upgrade_includes_versions() {
        if let CompatLevel::UpgradeRequired {
            from_major,
            to_major,
        } = matrix(5, 0, 2, 0)
        {
            assert_eq!(from_major, 2);
            assert_eq!(to_major, 5);
        }
    }

    #[test]
    fn patch_does_not_affect_matrix() {
        // We only consider major/minor; patch does not enter.
        assert_eq!(matrix(2, 0, 2, 0), CompatLevel::FullReadWrite);
    }

    #[test]
    fn at_grace_boundary_read_only() {
        // Exactly 1 major newer → ReadOnly.
        let v = matrix(2, 5, 1, 0);
        assert!(matches!(v, CompatLevel::ReadOnly { .. }));
    }

    #[test]
    fn just_past_grace_upgrade() {
        // 2 major newer → UpgradeRequired.
        let v = matrix(3, 0, 1, 0);
        assert!(matches!(v, CompatLevel::UpgradeRequired { .. }));
    }
}
