//! # apr monitor --log-rotation — File Rotation Budget
//!
//! `apr monitor` writes structured logs and rotates by size + count.
//! Constraints: max-bytes ≥ 1 MiB; keep-files ≥ 1; total budget = bytes
//! × keep, capped (default 10 GiB). This recipe builds the budget
//! validator.
//!
//! Demonstrates the **MON.6** recipe for PMAT-114 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MON-001 + log-rotation conventions
//!
//! Run with: cargo run --example cli_monitor_log_rotation_budget
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_FILE_BYTES: u64 = 1024 * 1024; // 1 MiB
const DEFAULT_TOTAL_CAP: u64 = 10 * 1024 * 1024 * 1024; // 10 GiB

#[derive(Debug, PartialEq)]
pub enum RotationVerdict {
    Ok { total_bytes: u64 },
    FileTooSmall,
    NoFilesToKeep,
    ExceedsTotalCap { total: u64, cap: u64 },
}

pub fn validate(max_bytes: u64, keep_files: u32, total_cap: u64) -> RotationVerdict {
    if max_bytes < MIN_FILE_BYTES {
        return RotationVerdict::FileTooSmall;
    }
    if keep_files == 0 {
        return RotationVerdict::NoFilesToKeep;
    }
    let total = max_bytes.saturating_mul(u64::from(keep_files));
    let cap = if total_cap == 0 {
        DEFAULT_TOTAL_CAP
    } else {
        total_cap
    };
    if total > cap {
        return RotationVerdict::ExceedsTotalCap { total, cap };
    }
    RotationVerdict::Ok { total_bytes: total }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_log_rotation_budget")?;

    let cases = [
        ("typical", 100 * 1024 * 1024, 10, 0),
        ("file too small", 1024, 10, 0),
        ("zero keep", 1024 * 1024, 0, 0),
        (
            "exceeds cap",
            10 * 1024 * 1024 * 1024,
            5,
            10 * 1024 * 1024 * 1024,
        ),
    ];
    for (label, b, k, cap) in cases {
        println!("{label:>16}  →  {:?}", validate(b, k, cap));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_config_passes() {
        // 100 MiB × 10 files = 1 GiB, well under default 10 GiB cap.
        let v = validate(100 * 1024 * 1024, 10, 0);
        assert!(matches!(v, RotationVerdict::Ok { .. }));
    }

    #[test]
    fn file_too_small_rejected() {
        // < 1 MiB rotates too fast and floods the directory.
        assert_eq!(validate(1024, 10, 0), RotationVerdict::FileTooSmall);
    }

    #[test]
    fn at_min_file_size_passes() {
        let v = validate(MIN_FILE_BYTES, 1, 0);
        assert!(matches!(v, RotationVerdict::Ok { .. }));
    }

    #[test]
    fn zero_keep_rejected() {
        assert_eq!(
            validate(MIN_FILE_BYTES, 0, 0),
            RotationVerdict::NoFilesToKeep
        );
    }

    #[test]
    fn exceeds_default_cap_rejected() {
        // 5 GiB × 5 files = 25 GiB > 10 GiB default.
        let v = validate(5 * 1024 * 1024 * 1024, 5, 0);
        assert!(matches!(v, RotationVerdict::ExceedsTotalCap { .. }));
    }

    #[test]
    fn explicit_cap_overrides_default() {
        // 1 GiB × 5 files = 5 GiB, with a 2 GiB explicit cap → reject.
        let v = validate(1024 * 1024 * 1024, 5, 2 * 1024 * 1024 * 1024);
        assert!(matches!(v, RotationVerdict::ExceedsTotalCap { .. }));
    }

    #[test]
    fn total_bytes_in_ok_matches_product() {
        if let RotationVerdict::Ok { total_bytes } = validate(2 * 1024 * 1024, 5, 0) {
            assert_eq!(total_bytes, 10 * 1024 * 1024);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn saturating_multiplication_no_overflow() {
        // Ensure max_bytes × keep doesn't panic on u64 overflow.
        let v = validate(u64::MAX / 2, u32::MAX, 0);
        assert!(matches!(v, RotationVerdict::ExceedsTotalCap { .. }));
    }
}
