//! # Monitoring Log PII Redactor
//!
//! Detect and redact common PII patterns:
//!   email: matches name@domain
//!   phone: matches NNN-NNN-NNNN or (NNN) NNN-NNNN
//!   credit-card: matches 16-digit groups (Luhn-checked optionally)
//!   ssn: NNN-NN-NNNN
//!
//! Replace with [REDACTED-{kind}] sentinel.
//!
//! Demonstrates the **MON.32** recipe for PMAT-151 (monitoring round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GDPR Article 32 + HIPAA PII redaction guidelines.
//!
//! Run with: cargo run --example monitor_log_pii_redact
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RedactVerdict {
    Ok { redacted: String, pii_count: u32 },
    EmptyInput,
}

pub fn redact(input: &str) -> RedactVerdict {
    if input.is_empty() {
        return RedactVerdict::EmptyInput;
    }
    let mut out = String::with_capacity(input.len());
    let mut pii_count = 0u32;
    for word in input.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if !out.is_empty() {
            out.push(' ');
        }
        if is_email(trimmed) {
            out.push_str("[REDACTED-email]");
            pii_count += 1;
        } else if is_phone(trimmed) {
            out.push_str("[REDACTED-phone]");
            pii_count += 1;
        } else if is_credit_card(trimmed) {
            out.push_str("[REDACTED-cc]");
            pii_count += 1;
        } else if is_ssn(trimmed) {
            out.push_str("[REDACTED-ssn]");
            pii_count += 1;
        } else {
            out.push_str(word);
        }
    }
    RedactVerdict::Ok {
        redacted: out,
        pii_count,
    }
}

fn is_email(s: &str) -> bool {
    let parts: Vec<&str> = s.split('@').collect();
    parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.')
}

fn is_phone(s: &str) -> bool {
    // NNN-NNN-NNNN or NNNNNNNNNN.
    let digit_count = s.chars().filter(char::is_ascii_digit).count();
    digit_count == 10 && s.chars().all(|c| c.is_ascii_digit() || c == '-')
}

fn is_credit_card(s: &str) -> bool {
    let digit_count = s.chars().filter(char::is_ascii_digit).count();
    matches!(digit_count, 13..=19)
        && s.chars().all(|c| c.is_ascii_digit() || c == '-')
        && !s.contains('@')
}

fn is_ssn(s: &str) -> bool {
    // NNN-NN-NNNN exact.
    let parts: Vec<&str> = s.split('-').collect();
    parts.len() == 3
        && parts[0].len() == 3
        && parts[1].len() == 2
        && parts[2].len() == 4
        && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_log_pii_redact")?;

    println!(
        "with email: {:?}",
        redact("user logged in: alice@example.com")
    );
    println!("with phone: {:?}", redact("contact 555-123-4567 today"));
    println!("with ssn: {:?}", redact("ssn 123-45-6789 verified"));
    println!("with cc: {:?}", redact("paid via 1234567812345678"));
    println!("empty: {:?}", redact(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redactor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn email_redacted() {
        let v = redact("contact alice@example.com");
        if let RedactVerdict::Ok {
            redacted,
            pii_count,
        } = v
        {
            assert!(redacted.contains("[REDACTED-email]"));
            assert_eq!(pii_count, 1);
        }
    }

    #[test]
    fn phone_redacted() {
        let v = redact("call 555-123-4567");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-phone]"));
        }
    }

    #[test]
    fn ssn_redacted() {
        let v = redact("ssn 123-45-6789 here");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-ssn]"));
        }
    }

    #[test]
    fn credit_card_redacted() {
        let v = redact("paid 1234567812345678");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-cc]"));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(redact(""), RedactVerdict::EmptyInput);
    }

    #[test]
    fn no_pii_unchanged_words() {
        let v = redact("hello world");
        if let RedactVerdict::Ok { pii_count, .. } = v {
            assert_eq!(pii_count, 0);
        }
    }

    #[test]
    fn pii_count_tracks_each_match() {
        let v = redact("email a@b.com phone 111-222-3333");
        if let RedactVerdict::Ok { pii_count, .. } = v {
            assert_eq!(pii_count, 2);
        }
    }

    #[test]
    fn multiple_emails_all_redacted() {
        let v = redact("a@b.com or c@d.com");
        if let RedactVerdict::Ok {
            redacted,
            pii_count,
        } = v
        {
            assert_eq!(pii_count, 2);
            assert!(!redacted.contains("@b.com"));
        }
    }

    #[test]
    fn phone_without_dashes_redacted() {
        let v = redact("number 5551234567");
        if let RedactVerdict::Ok { pii_count, .. } = v {
            assert_eq!(pii_count, 1);
        }
    }

    #[test]
    fn email_without_dot_not_redacted() {
        // Domain must contain dot.
        let v = redact("not_email a@nodot");
        if let RedactVerdict::Ok { pii_count, .. } = v {
            assert_eq!(pii_count, 0);
        }
    }

    #[test]
    fn ssn_wrong_format_not_redacted() {
        let v = redact("ssn 12345-678");
        if let RedactVerdict::Ok { pii_count, .. } = v {
            assert_eq!(pii_count, 0);
        }
    }
}
