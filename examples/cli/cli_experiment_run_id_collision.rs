//! # apr experiment — Run ID Collision Detector
//!
//! Run IDs use `<timestamp>-<short-hash>` to enable parallel runs
//! without collision. Constraints: timestamp = `YYYYMMDDHHMMSS` (14
//! digits); short-hash = 8 hex chars; collision requires both halves
//! match. This recipe builds the validator + collision checker.
//!
//! Demonstrates the **EXP.4** recipe for PMAT-118 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + ULID/UUID short-form conventions
//!
//! Run with: cargo run --example cli_experiment_run_id_collision
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum IdVerdict {
    Ok,
    InvalidFormat,
    BadTimestamp,
    BadHash,
}

pub fn validate(id: &str) -> IdVerdict {
    let Some((ts, hash)) = id.split_once('-') else {
        return IdVerdict::InvalidFormat;
    };
    if ts.len() != 14 || !ts.chars().all(|c| c.is_ascii_digit()) {
        return IdVerdict::BadTimestamp;
    }
    if hash.len() != 8 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return IdVerdict::BadHash;
    }
    IdVerdict::Ok
}

pub fn first_collision<'a>(ids: &'a [&'a str]) -> Option<&'a str> {
    let mut seen: HashSet<&str> = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Some(*id);
        }
    }
    None
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_run_id_collision")?;

    let cases = [
        "20260506093000-deadbeef",
        "20260506093000-DEADBEEF",
        "bad-format",
        "shortts-deadbeef",
        "20260506093000-zzzzzzzz",
    ];
    for c in cases {
        println!("{c:>30}  →  {:?}", validate(c));
    }

    let ids = [
        "20260506093000-aaaaaaaa",
        "20260506093001-bbbbbbbb",
        "20260506093000-aaaaaaaa", // duplicate
    ];
    println!("collision: {:?}", first_collision(&ids));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_id_passes() {
        assert_eq!(validate("20260506093000-deadbeef"), IdVerdict::Ok);
    }

    #[test]
    fn uppercase_hex_passes() {
        assert_eq!(validate("20260506093000-DEADBEEF"), IdVerdict::Ok);
    }

    #[test]
    fn mixed_case_hex_passes() {
        assert_eq!(validate("20260506093000-DeadBeef"), IdVerdict::Ok);
    }

    #[test]
    fn missing_separator_invalid() {
        assert_eq!(validate("20260506093000deadbeef"), IdVerdict::InvalidFormat);
    }

    #[test]
    fn short_timestamp_rejected() {
        assert_eq!(validate("12345-deadbeef"), IdVerdict::BadTimestamp);
    }

    #[test]
    fn non_digit_timestamp_rejected() {
        assert_eq!(validate("abcd0506093000-deadbeef"), IdVerdict::BadTimestamp);
    }

    #[test]
    fn short_hash_rejected() {
        assert_eq!(validate("20260506093000-dead"), IdVerdict::BadHash);
    }

    #[test]
    fn non_hex_hash_rejected() {
        // 'z' is not a hex digit.
        assert_eq!(validate("20260506093000-deadbezz"), IdVerdict::BadHash);
    }

    #[test]
    fn collision_detected_in_list() {
        let ids = ["a", "b", "a"];
        assert_eq!(first_collision(&ids), Some("a"));
    }

    #[test]
    fn no_collision_returns_none() {
        let ids = ["a", "b", "c"];
        assert!(first_collision(&ids).is_none());
    }
}
