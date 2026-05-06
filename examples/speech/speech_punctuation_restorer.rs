//! # Speech Punctuation Restorer
//!
//! Use token gap durations to insert punctuation:
//!   gap > 0.5s → period
//!   gap 0.2-0.5s → comma
//!   gap < 0.2s → no punctuation
//!
//! Plus capitalize first letter after period.
//!
//! Demonstrates the **SPEECH.9** recipe for PMAT-149 (speech round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: deepmultilingualpunctuation gap-based heuristic.
//!
//! Run with: cargo run --example speech_punctuation_restorer
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PERIOD_GAP_SECS: f64 = 0.5;
const COMMA_GAP_SECS: f64 = 0.2;

#[derive(Debug, PartialEq)]
pub enum PunctVerdict {
    Ok(String),
    EmptyTokens,
    GapsLengthMismatch,
    InvalidGap,
}

pub fn restore(tokens: &[&str], gaps_after: &[f64]) -> PunctVerdict {
    if tokens.is_empty() {
        return PunctVerdict::EmptyTokens;
    }
    if gaps_after.len() != tokens.len() {
        return PunctVerdict::GapsLengthMismatch;
    }
    if gaps_after.iter().any(|g| !g.is_finite() || *g < 0.0) {
        return PunctVerdict::InvalidGap;
    }
    let mut out = String::new();
    let mut capitalize_next = true;
    for (i, &tok) in tokens.iter().enumerate() {
        let mut t = tok.to_string();
        if capitalize_next {
            if let Some(first) = t.chars().next() {
                let upper: String = first.to_uppercase().collect();
                t = format!("{upper}{}", &t[first.len_utf8()..]);
            }
            capitalize_next = false;
        }
        out.push_str(&t);
        let gap = gaps_after[i];
        if gap >= PERIOD_GAP_SECS {
            out.push('.');
            capitalize_next = true;
        } else if gap >= COMMA_GAP_SECS {
            out.push(',');
        }
        if i < tokens.len() - 1 {
            out.push(' ');
        }
    }
    PunctVerdict::Ok(out)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_punctuation_restorer")?;

    let tokens = ["hello", "world", "this", "is", "a", "test"];
    let gaps = [0.6, 0.1, 0.3, 0.1, 0.1, 0.0];
    println!("typical: {:?}", restore(&tokens, &gaps));

    println!("empty: {:?}", restore(&[], &[]));
    println!("mismatch: {:?}", restore(&["a"], &[0.5, 0.1]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn period_inserted_on_long_gap() {
        let tokens = ["hello", "world"];
        let gaps = [0.6, 0.0];
        if let PunctVerdict::Ok(s) = restore(&tokens, &gaps) {
            assert!(s.contains("Hello.") || s.contains("hello."));
        }
    }

    #[test]
    fn comma_inserted_on_medium_gap() {
        let tokens = ["a", "b"];
        let gaps = [0.3, 0.0];
        if let PunctVerdict::Ok(s) = restore(&tokens, &gaps) {
            assert!(s.contains(','));
        }
    }

    #[test]
    fn no_punct_on_short_gap() {
        let tokens = ["a", "b"];
        let gaps = [0.1, 0.0];
        if let PunctVerdict::Ok(s) = restore(&tokens, &gaps) {
            assert!(!s.contains('.'));
            assert!(!s.contains(','));
        }
    }

    #[test]
    fn capitalize_first_word() {
        let v = restore(&["hello"], &[0.0]);
        if let PunctVerdict::Ok(s) = v {
            assert!(s.starts_with('H'));
        }
    }

    #[test]
    fn capitalize_after_period() {
        let tokens = ["yes", "no"];
        let gaps = [0.6, 0.0];
        if let PunctVerdict::Ok(s) = restore(&tokens, &gaps) {
            assert!(s.contains("No"));
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(restore(&[], &[]), PunctVerdict::EmptyTokens);
    }

    #[test]
    fn mismatch_rejected() {
        let v = restore(&["a"], &[0.5, 0.1]);
        assert_eq!(v, PunctVerdict::GapsLengthMismatch);
    }

    #[test]
    fn nan_gap_rejected() {
        let v = restore(&["a", "b"], &[f64::NAN, 0.0]);
        assert_eq!(v, PunctVerdict::InvalidGap);
    }

    #[test]
    fn negative_gap_rejected() {
        let v = restore(&["a", "b"], &[-0.1, 0.0]);
        assert_eq!(v, PunctVerdict::InvalidGap);
    }

    #[test]
    fn boundary_at_period_threshold() {
        let v = restore(&["a", "b"], &[PERIOD_GAP_SECS, 0.0]);
        if let PunctVerdict::Ok(s) = v {
            assert!(s.contains('.'));
        }
    }

    #[test]
    fn just_below_period_uses_comma() {
        let v = restore(&["a", "b"], &[PERIOD_GAP_SECS - 0.01, 0.0]);
        if let PunctVerdict::Ok(s) = v {
            assert!(s.contains(','));
            assert!(!s.contains('.'));
        }
    }

    #[test]
    fn unicode_capitalize() {
        let v = restore(&["école"], &[0.0]);
        if let PunctVerdict::Ok(s) = v {
            assert!(s.starts_with('É'));
        }
    }
}
