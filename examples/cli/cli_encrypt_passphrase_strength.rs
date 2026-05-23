//! # apr encrypt — Passphrase Strength Validator
//!
//! `apr encrypt` accepts a passphrase via stdin (when `--key-file` is
//! omitted). The passphrase is fed to BLAKE3's `derive_key`. This recipe
//! builds a strength validator that rejects weak passphrases at the
//! boundary: minimum 12 chars, ≥ 3 character classes, no repeated runs
//! of length ≥ 4.
//!
//! Demonstrates the **ENCRYPT.4** recipe for PMAT-103 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + NIST SP 800-63B (passphrase guidelines)
//!
//! Run with: cargo run --example cli_encrypt_passphrase_strength
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StrengthVerdict {
    Strong,
    TooShort { observed: usize, required: usize },
    InsufficientClasses { observed: usize, required: usize },
    LongRepeat { run: String },
}

const MIN_LEN: usize = 12;
const MIN_CLASSES: usize = 3;
const MAX_RUN: usize = 3;

fn class_count(s: &str) -> usize {
    let has_lower = s.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = s.chars().any(|c| c.is_ascii_uppercase());
    let has_digit = s.chars().any(|c| c.is_ascii_digit());
    let has_punct = s.chars().any(|c| c.is_ascii_punctuation() || c == ' ');
    [has_lower, has_upper, has_digit, has_punct]
        .iter()
        .filter(|&&b| b)
        .count()
}

fn longest_run(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut max_run: Vec<char> = Vec::new();
    let mut cur_run: Vec<char> = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        if i == 0 || *c == chars[i - 1] {
            cur_run.push(*c);
        } else {
            if cur_run.len() > max_run.len() {
                max_run = cur_run.clone();
            }
            cur_run = vec![*c];
        }
    }
    if cur_run.len() > max_run.len() {
        max_run = cur_run;
    }
    if max_run.len() > MAX_RUN {
        Some(max_run.into_iter().collect())
    } else {
        None
    }
}

pub fn validate(passphrase: &str) -> StrengthVerdict {
    if passphrase.len() < MIN_LEN {
        return StrengthVerdict::TooShort {
            observed: passphrase.len(),
            required: MIN_LEN,
        };
    }
    let classes = class_count(passphrase);
    if classes < MIN_CLASSES {
        return StrengthVerdict::InsufficientClasses {
            observed: classes,
            required: MIN_CLASSES,
        };
    }
    if let Some(run) = longest_run(passphrase) {
        return StrengthVerdict::LongRepeat { run };
    }
    StrengthVerdict::Strong
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_passphrase_strength")?;

    for p in [
        "correct horse battery staple 9!",
        "short",
        "alllowercase only",
        "aaaaaa AAAAAA 11111",
        "Tr0ub4dor&3",
    ] {
        println!("{p:>40}  →  {:?}", validate(p));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn strong_passphrase_passes() {
        assert_eq!(validate("Mix3d-bag of words!"), StrengthVerdict::Strong);
    }

    #[test]
    fn short_passphrase_rejected() {
        let v = validate("Tr0ub4dor");
        assert!(matches!(v, StrengthVerdict::TooShort { .. }));
    }

    #[test]
    fn insufficient_classes_rejected() {
        // All-lowercase + space → 2 classes; below the 3-class floor.
        let v = validate("all lowercase only");
        assert!(matches!(v, StrengthVerdict::InsufficientClasses { .. }));
    }

    #[test]
    fn long_repeat_rejected() {
        // Run of 6 same chars triggers LongRepeat.
        let v = validate("aaaaaa AAA 12!");
        assert!(matches!(v, StrengthVerdict::LongRepeat { .. }));
    }

    #[test]
    fn boundary_at_exactly_12_chars_passes() {
        // 12 chars, 4 classes, no run ≥ 4 — must pass.
        assert_eq!(validate("Ab1!Cd2@Ef3#"), StrengthVerdict::Strong);
    }

    #[test]
    fn class_count_correct() {
        assert_eq!(class_count("abc"), 1);
        assert_eq!(class_count("aB"), 2);
        assert_eq!(class_count("aB1"), 3);
        assert_eq!(class_count("aB1!"), 4);
    }

    #[test]
    fn longest_run_only_flags_above_threshold() {
        assert!(longest_run("abc").is_none());
        assert!(longest_run("aaab").is_none()); // run of 3 — at threshold, allowed
        assert_eq!(longest_run("aaaabc"), Some("aaaa".into()));
    }
}
