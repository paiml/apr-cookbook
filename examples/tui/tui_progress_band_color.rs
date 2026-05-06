//! # TUI Progress Bar Color Band
//!
//! Map progress percent to color band (`red`, `yellow`, `green`).
//! Useful for status bars and dashboards.
//!
//! Demonstrates the **TUI.78** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: traffic-light heuristic; Don Norman, Design of Everyday
//!  Things ch.3 (signifiers).
//!
//! Run with: cargo run --example tui_progress_band_color
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BandColor {
    Red,
    Yellow,
    Green,
}

#[derive(Debug, PartialEq)]
pub enum BandVerdict {
    Ok { color: BandColor, label: String },
    InvalidConfig,
}

pub fn classify(percent: u32) -> BandVerdict {
    if percent > 100 {
        return BandVerdict::InvalidConfig;
    }
    let (color, label) = if percent < 30 {
        (BandColor::Red, "low")
    } else if percent < 70 {
        (BandColor::Yellow, "mid")
    } else {
        (BandColor::Green, "high")
    };
    BandVerdict::Ok {
        color,
        label: label.to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_band_color")?;

    println!("low: {:?}", classify(10));
    println!("mid: {:?}", classify(50));
    println!("high: {:?}", classify(90));
    println!("invalid: {:?}", classify(150));
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
    fn zero_percent_red() {
        let v = classify(0);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Red);
        }
    }

    #[test]
    fn fifty_percent_yellow() {
        let v = classify(50);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Yellow);
        }
    }

    #[test]
    fn hundred_percent_green() {
        let v = classify(100);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Green);
        }
    }

    #[test]
    fn invalid_above_hundred() {
        assert_eq!(classify(101), BandVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_30_yellow() {
        let v = classify(30);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Yellow);
        }
    }

    #[test]
    fn boundary_29_red() {
        let v = classify(29);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Red);
        }
    }

    #[test]
    fn boundary_70_green() {
        let v = classify(70);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Green);
        }
    }

    #[test]
    fn boundary_69_yellow() {
        let v = classify(69);
        if let BandVerdict::Ok { color, .. } = v {
            assert_eq!(color, BandColor::Yellow);
        }
    }

    #[test]
    fn label_low_for_red() {
        let v = classify(10);
        if let BandVerdict::Ok { label, .. } = v {
            assert_eq!(label, "low");
        }
    }

    #[test]
    fn label_mid_for_yellow() {
        let v = classify(50);
        if let BandVerdict::Ok { label, .. } = v {
            assert_eq!(label, "mid");
        }
    }

    #[test]
    fn label_high_for_green() {
        let v = classify(90);
        if let BandVerdict::Ok { label, .. } = v {
            assert_eq!(label, "high");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = classify(50);
        let r2 = classify(50);
        assert_eq!(r1, r2);
    }
}
