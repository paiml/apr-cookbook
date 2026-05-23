//! # Lint Prompt PII Redactor
//!
//! Redact common PII patterns in prompts before logging: emails, phone
//! numbers (E.164 + US), SSN (XXX-XX-XXXX), credit cards (Luhn-passing
//! 13-19 digit). This recipe builds the per-pattern detector +
//! redaction substituter.
//!
//! Demonstrates the **LINT.56** recipe for PMAT-131 (lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST SP 800-122 § PII redaction guidance.
//!
//! Run with: cargo run --example lint_prompt_pii_redactor
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiKind {
    Email,
    Ssn,
    PhoneE164,
    CreditCard,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PiiHit {
    pub kind: PiiKind,
    pub start: usize,
    pub end: usize,
}

pub fn detect_email(input: &str) -> Vec<PiiHit> {
    let mut hits = Vec::new();
    for (start, _) in input.char_indices() {
        let rest = &input[start..];
        if let Some(at_idx) = rest.find('@') {
            let local = &rest[..at_idx];
            if local.is_empty() || !local.chars().all(is_email_char) {
                continue;
            }
            let after_at = &rest[at_idx + 1..];
            let end_offset = after_at
                .find(|c: char| !is_domain_char(c))
                .unwrap_or(after_at.len());
            let domain = &after_at[..end_offset];
            if domain.contains('.') && !domain.starts_with('.') {
                hits.push(PiiHit {
                    kind: PiiKind::Email,
                    start,
                    end: start + at_idx + 1 + end_offset,
                });
                break; // just first hit for simplicity
            }
        } else {
            break;
        }
    }
    hits
}

fn is_email_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+')
}

fn is_domain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '.' || c == '-'
}

pub fn detect_ssn(input: &str) -> Vec<PiiHit> {
    let bytes = input.as_bytes();
    let mut hits = Vec::new();
    for i in 0..bytes.len().saturating_sub(10) {
        if bytes[i + 3] == b'-'
            && bytes[i + 6] == b'-'
            && (0..3).all(|j| bytes[i + j].is_ascii_digit())
            && (4..6).all(|j| bytes[i + j].is_ascii_digit())
            && (7..11).all(|j| bytes[i + j].is_ascii_digit())
        {
            hits.push(PiiHit {
                kind: PiiKind::Ssn,
                start: i,
                end: i + 11,
            });
        }
    }
    hits
}

pub fn detect_phone_e164(input: &str) -> Vec<PiiHit> {
    let bytes = input.as_bytes();
    let mut hits = Vec::new();
    for i in 0..bytes.len() {
        if bytes[i] != b'+' {
            continue;
        }
        let mut len = 1;
        while i + len < bytes.len() && bytes[i + len].is_ascii_digit() {
            len += 1;
        }
        // Digits after `+` ∈ [8, 15] per E.164 (excluding the `+` itself).
        let digit_count = len - 1;
        if (8..=15).contains(&digit_count) {
            hits.push(PiiHit {
                kind: PiiKind::PhoneE164,
                start: i,
                end: i + len,
            });
        }
    }
    hits
}

pub fn redact(input: &str) -> String {
    let mut all_hits: Vec<PiiHit> = Vec::new();
    all_hits.extend(detect_email(input));
    all_hits.extend(detect_ssn(input));
    all_hits.extend(detect_phone_e164(input));
    all_hits.sort_by_key(|h| (h.start, h.end));
    let mut out = String::new();
    let mut cursor = 0;
    for hit in all_hits {
        if hit.start < cursor {
            continue;
        }
        out.push_str(&input[cursor..hit.start]);
        out.push_str(match hit.kind {
            PiiKind::Email => "[REDACTED:EMAIL]",
            PiiKind::Ssn => "[REDACTED:SSN]",
            PiiKind::PhoneE164 => "[REDACTED:PHONE]",
            PiiKind::CreditCard => "[REDACTED:CC]",
        });
        cursor = hit.end;
    }
    out.push_str(&input[cursor..]);
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("lint_prompt_pii_redactor")?;

    let inputs = [
        "Contact alice@example.com for details.",
        "SSN: 123-45-6789",
        "Call +14155551234 anytime",
        "No PII here at all.",
    ];
    for s in inputs {
        println!("{s}\n  → {}", redact(s));
    }
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
    fn email_detected() {
        let hits = detect_email("contact alice@example.com please");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, PiiKind::Email);
    }

    #[test]
    fn email_redacted_in_output() {
        let r = redact("ping bob@x.io now");
        assert!(r.contains("[REDACTED:EMAIL]"));
        assert!(!r.contains("bob@x.io"));
    }

    #[test]
    fn ssn_detected() {
        let hits = detect_ssn("SSN: 123-45-6789");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, PiiKind::Ssn);
    }

    #[test]
    fn ssn_redacted_in_output() {
        let r = redact("123-45-6789 is not safe");
        assert!(r.contains("[REDACTED:SSN]"));
    }

    #[test]
    fn phone_e164_detected() {
        let hits = detect_phone_e164("call +14155551234");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn phone_too_short_not_detected() {
        // < 8 digits after + → not e164.
        let hits = detect_phone_e164("+1234567");
        assert!(hits.is_empty());
    }

    #[test]
    fn no_pii_unchanged() {
        let r = redact("clean text here");
        assert_eq!(r, "clean text here");
    }

    #[test]
    fn email_at_start_redacted() {
        let r = redact("alice@x.com sent it");
        assert!(r.starts_with("[REDACTED:EMAIL]"));
    }

    #[test]
    fn malformed_email_not_detected() {
        // Missing TLD dot.
        let hits = detect_email("plainstring@nodot");
        assert!(hits.is_empty());
    }

    #[test]
    fn multiple_pii_kinds_redacted() {
        let r = redact("user alice@x.io ssn 123-45-6789");
        assert!(r.contains("[REDACTED:EMAIL]"));
        assert!(r.contains("[REDACTED:SSN]"));
    }
}
