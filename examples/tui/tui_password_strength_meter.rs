//! # TUI Password Strength Meter
//!
//! Estimate password strength as discrete bars (0..=4) using a
//! simple Shannon-entropy approximation: `len * log2(charset_size)`,
//! then map to bands. Returns bar count + estimated bits.
//!
//! Demonstrates the **TUI.63** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST SP 800-63B Appendix A; Shannon, A Mathematical
//!  Theory of Communication (1948).
//!
//! Run with: cargo run --example tui_password_strength_meter
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StrengthVerdict {
    Ok { bars: u32, bits: f64 },
    InvalidConfig,
}

pub fn classify(password: &str) -> StrengthVerdict {
    if password.is_empty() {
        return StrengthVerdict::InvalidConfig;
    }
    let len = password.chars().count();
    let mut has_lower = false;
    let mut has_upper = false;
    let mut has_digit = false;
    let mut has_symbol = false;
    for c in password.chars() {
        if c.is_ascii_lowercase() {
            has_lower = true;
        } else if c.is_ascii_uppercase() {
            has_upper = true;
        } else if c.is_ascii_digit() {
            has_digit = true;
        } else {
            has_symbol = true;
        }
    }
    let mut charset = 0u32;
    if has_lower {
        charset += 26;
    }
    if has_upper {
        charset += 26;
    }
    if has_digit {
        charset += 10;
    }
    if has_symbol {
        charset += 32;
    }
    let bits = if charset == 0 {
        0.0
    } else {
        (len as f64) * f64::from(charset).log2()
    };
    let bars = if bits < 28.0 {
        0
    } else if bits < 36.0 {
        1
    } else if bits < 60.0 {
        2
    } else if bits < 90.0 {
        3
    } else {
        4
    };
    StrengthVerdict::Ok { bars, bits }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_password_strength_meter")?;

    println!("weak: {:?}", classify("abc"));
    println!("medium: {:?}", classify("Hello123"));
    println!("strong: {:?}", classify("XyZ!23#abc4_def"));
    println!("invalid: {:?}", classify(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn very_short_is_zero_bars() {
        let v = classify("ab");
        if let StrengthVerdict::Ok { bars, .. } = v {
            assert_eq!(bars, 0);
        }
    }

    #[test]
    fn long_complex_max_bars() {
        let v = classify("Xy7!Zw2@qr5#st9$");
        if let StrengthVerdict::Ok { bars, .. } = v {
            assert_eq!(bars, 4);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(classify(""), StrengthVerdict::InvalidConfig);
    }

    #[test]
    fn bars_in_zero_to_four() {
        let v = classify("hello");
        if let StrengthVerdict::Ok { bars, .. } = v {
            assert!(bars <= 4);
        }
    }

    #[test]
    fn bits_nonneg() {
        let v = classify("abc");
        if let StrengthVerdict::Ok { bits, .. } = v {
            assert!(bits >= 0.0);
        }
    }

    #[test]
    fn longer_more_bits_same_charset() {
        let short = classify("aaa");
        let long = classify("aaaaaaaa");
        if let (StrengthVerdict::Ok { bits: s, .. }, StrengthVerdict::Ok { bits: l, .. }) =
            (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn more_charset_more_bits_same_len() {
        let lower = classify("abcdef");
        let mixed = classify("aBcDeF");
        if let (StrengthVerdict::Ok { bits: l, .. }, StrengthVerdict::Ok { bits: m, .. }) =
            (lower, mixed)
        {
            assert!(m > l);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = classify("Hello123");
        let r2 = classify("Hello123");
        assert_eq!(r1, r2);
    }

    #[test]
    fn symbol_increases_charset() {
        let no_sym = classify("Hello12");
        let sym = classify("Hello1!");
        if let (StrengthVerdict::Ok { bits: n, .. }, StrengthVerdict::Ok { bits: s, .. }) =
            (no_sym, sym)
        {
            assert!(s > n);
        }
    }

    #[test]
    fn unicode_treated_as_symbol() {
        let v = classify("café");
        if let StrengthVerdict::Ok { bits, .. } = v {
            assert!(bits > 0.0);
        }
    }

    #[test]
    fn one_char_low_bars() {
        let v = classify("a");
        if let StrengthVerdict::Ok { bars, .. } = v {
            assert_eq!(bars, 0);
        }
    }
}
