//! # Monitoring Log Correlation-ID Validator
//!
//! Every log line in a request's path must carry the same correlation
//! ID (UUID v4 typical). Validator checks:
//!   - All log lines have the field
//!   - All values match
//!   - Format is valid UUID
//!
//! Demonstrates the **MON.28** recipe for PMAT-147 (monitoring round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenTelemetry trace_id propagation conventions.
//!
//! Run with: cargo run --example monitor_log_correlation
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CorrelationVerdict {
    AllConsistent { id: String },
    MissingFromLine { line_index: usize },
    Mismatch { expected: String, found: String },
    InvalidUuidFormat { id: String },
    NoLines,
}

const UUID_LEN: usize = 36;

pub fn validate(log_correlation_ids: &[Option<&str>]) -> CorrelationVerdict {
    if log_correlation_ids.is_empty() {
        return CorrelationVerdict::NoLines;
    }
    let first = match log_correlation_ids[0] {
        Some(id) => id.to_string(),
        None => return CorrelationVerdict::MissingFromLine { line_index: 0 },
    };
    if !is_valid_uuid(&first) {
        return CorrelationVerdict::InvalidUuidFormat { id: first };
    }
    for (i, id) in log_correlation_ids.iter().enumerate().skip(1) {
        match id {
            Some(s) => {
                if *s != first {
                    return CorrelationVerdict::Mismatch {
                        expected: first,
                        found: (*s).to_string(),
                    };
                }
            }
            None => return CorrelationVerdict::MissingFromLine { line_index: i },
        }
    }
    CorrelationVerdict::AllConsistent { id: first }
}

fn is_valid_uuid(s: &str) -> bool {
    if s.len() != UUID_LEN {
        return false;
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if matches!(i, 8 | 13 | 18 | 23) {
            if c != '-' {
                return false;
            }
        } else if !c.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_log_correlation")?;

    let id = "550e8400-e29b-41d4-a716-446655440000";
    println!(
        "all consistent: {:?}",
        validate(&[Some(id), Some(id), Some(id)])
    );
    println!(
        "missing line 1: {:?}",
        validate(&[Some(id), None, Some(id)])
    );
    println!(
        "mismatch: {:?}",
        validate(&[Some(id), Some("11111111-1111-1111-1111-111111111111")])
    );
    println!("invalid format: {:?}", validate(&[Some("not-a-uuid")]));
    println!("no lines: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_UUID: &str = "550e8400-e29b-41d4-a716-446655440000";

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_match_consistent() {
        let v = validate(&[Some(VALID_UUID), Some(VALID_UUID)]);
        if let CorrelationVerdict::AllConsistent { id } = v {
            assert_eq!(id, VALID_UUID);
        }
    }

    #[test]
    fn missing_from_first_line() {
        let v = validate(&[None, Some(VALID_UUID)]);
        assert_eq!(v, CorrelationVerdict::MissingFromLine { line_index: 0 });
    }

    #[test]
    fn missing_from_middle_line() {
        let v = validate(&[Some(VALID_UUID), None, Some(VALID_UUID)]);
        assert_eq!(v, CorrelationVerdict::MissingFromLine { line_index: 1 });
    }

    #[test]
    fn mismatched_ids_rejected() {
        let other = "11111111-1111-1111-1111-111111111111";
        let v = validate(&[Some(VALID_UUID), Some(other)]);
        assert!(matches!(v, CorrelationVerdict::Mismatch { .. }));
    }

    #[test]
    fn invalid_format_rejected() {
        let v = validate(&[Some("not-a-uuid")]);
        assert!(matches!(v, CorrelationVerdict::InvalidUuidFormat { .. }));
    }

    #[test]
    fn no_lines_rejected() {
        assert_eq!(validate(&[]), CorrelationVerdict::NoLines);
    }

    #[test]
    fn wrong_length_rejected() {
        // Too short.
        let v = validate(&[Some("12345")]);
        assert!(matches!(v, CorrelationVerdict::InvalidUuidFormat { .. }));
    }

    #[test]
    fn missing_dash_rejected() {
        let bad = "550e8400e29b-41d4-a716-446655440000";
        let v = validate(&[Some(bad)]);
        assert!(matches!(v, CorrelationVerdict::InvalidUuidFormat { .. }));
    }

    #[test]
    fn non_hex_char_rejected() {
        let bad = "550e8400-e29b-41d4-a716-zzzzzzzzzzzz";
        let v = validate(&[Some(bad)]);
        assert!(matches!(v, CorrelationVerdict::InvalidUuidFormat { .. }));
    }

    #[test]
    fn single_valid_line_accepted() {
        let v = validate(&[Some(VALID_UUID)]);
        assert!(matches!(v, CorrelationVerdict::AllConsistent { .. }));
    }
}
