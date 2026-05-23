//! # TUI Color Contrast Pass (WCAG)
//!
//! Compute WCAG 2.2 contrast ratio between fg and bg given relative
//! luminance values [0,1]; classify into AAA / AA-Large / AA / Fail.
//!
//! Demonstrates the **TUI.59** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WCAG 2.2 §1.4.3 (contrast minimum) and §1.4.6 (enhanced).
//!
//! Run with: cargo run --example tui_color_contrast_pass
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ContrastClass {
    Aaa,
    AaLarge,
    Aa,
    Fail,
}

#[derive(Debug, PartialEq)]
pub enum ContrastVerdict {
    Ok {
        ratio: f64,
        classification: ContrastClass,
    },
    InvalidConfig,
}

pub fn classify(fg_luminance: f64, bg_luminance: f64) -> ContrastVerdict {
    if !(0.0..=1.0).contains(&fg_luminance) || !(0.0..=1.0).contains(&bg_luminance) {
        return ContrastVerdict::InvalidConfig;
    }
    let lighter = fg_luminance.max(bg_luminance);
    let darker = fg_luminance.min(bg_luminance);
    let ratio = (lighter + 0.05) / (darker + 0.05);
    let classification = if ratio >= 7.0 {
        ContrastClass::Aaa
    } else if ratio >= 4.5 {
        ContrastClass::Aa
    } else if ratio >= 3.0 {
        ContrastClass::AaLarge
    } else {
        ContrastClass::Fail
    };
    ContrastVerdict::Ok {
        ratio,
        classification,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_color_contrast_pass")?;

    println!("white on black: {:?}", classify(1.0, 0.0));
    println!("white on gray: {:?}", classify(1.0, 0.5));
    println!("low contrast: {:?}", classify(0.6, 0.5));
    println!("invalid: {:?}", classify(2.0, 0.5));
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
    fn black_white_aaa() {
        let v = classify(1.0, 0.0);
        if let ContrastVerdict::Ok {
            classification,
            ratio,
        } = v
        {
            assert_eq!(classification, ContrastClass::Aaa);
            assert!((ratio - 21.0).abs() < 0.01);
        }
    }

    #[test]
    fn low_contrast_fails() {
        let v = classify(0.55, 0.50);
        if let ContrastVerdict::Ok { classification, .. } = v {
            assert_eq!(classification, ContrastClass::Fail);
        }
    }

    #[test]
    fn ratio_is_symmetric() {
        let a = classify(0.2, 0.8);
        let b = classify(0.8, 0.2);
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_negative() {
        assert_eq!(classify(-0.1, 0.5), ContrastVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_above_one() {
        assert_eq!(classify(1.5, 0.5), ContrastVerdict::InvalidConfig);
    }

    #[test]
    fn equal_luminance_ratio_one() {
        let v = classify(0.5, 0.5);
        if let ContrastVerdict::Ok {
            classification,
            ratio,
        } = v
        {
            assert!((ratio - 1.0).abs() < 1e-9);
            assert_eq!(classification, ContrastClass::Fail);
        }
    }

    #[test]
    fn aa_threshold_45() {
        // L1 = 0.6, L2 = 0.05 → ratio = (0.6+0.05)/(0.05+0.05) = 6.5 → AA.
        let v = classify(0.6, 0.05);
        if let ContrastVerdict::Ok { classification, .. } = v {
            assert_eq!(classification, ContrastClass::Aa);
        }
    }

    #[test]
    fn aa_large_threshold_30() {
        // ratio between 3 and 4.5 → AaLarge.
        // L1=0.4, L2=0.05 → (0.45)/(0.10) = 4.5 → exactly AA boundary.
        // Use a slightly lower one: L1=0.35, L2=0.05 → 0.40/0.10 = 4.0 → AA.
        // Try L1=0.30, L2=0.05 → 0.35/0.10 = 3.5 → AaLarge.
        let v = classify(0.30, 0.05);
        if let ContrastVerdict::Ok { classification, .. } = v {
            assert_eq!(classification, ContrastClass::AaLarge);
        }
    }

    #[test]
    fn deterministic() {
        let a = classify(0.7, 0.2);
        let b = classify(0.7, 0.2);
        assert_eq!(a, b);
    }

    #[test]
    fn ratio_at_least_one() {
        let v = classify(0.5, 0.5);
        if let ContrastVerdict::Ok { ratio, .. } = v {
            assert!(ratio >= 1.0);
        }
    }

    #[test]
    fn extreme_zero_one_max_ratio() {
        let v = classify(0.0, 1.0);
        if let ContrastVerdict::Ok { ratio, .. } = v {
            assert!(ratio > 20.0);
        }
    }
}
