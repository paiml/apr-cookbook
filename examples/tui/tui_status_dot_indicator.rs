//! # TUI Status Dot Indicator
//!
//! Pick the colored dot glyph for a service health value
//! (0..=100): ●green ≥ 90, ●yellow ≥ 50, ●red < 50, ●gray
//! when no data. Returns glyph + ANSI color code.
//!
//! Demonstrates the **TUI.51** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: traffic-light health indicators (GitHub status page).
//!
//! Run with: cargo run --example tui_status_dot_indicator
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DotVerdict {
    Pick { glyph: char, ansi_color: u32 },
    Unknown,
    InvalidHealth,
}

pub fn pick(health_pct: Option<f64>) -> DotVerdict {
    let Some(h) = health_pct else {
        return DotVerdict::Unknown;
    };
    if !h.is_finite() || !(0.0..=100.0).contains(&h) {
        return DotVerdict::InvalidHealth;
    }
    let (glyph, ansi_color) = if h >= 90.0 {
        ('●', 32) // green
    } else if h >= 50.0 {
        ('●', 33) // yellow
    } else {
        ('●', 31) // red
    };
    DotVerdict::Pick { glyph, ansi_color }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_status_dot_indicator")?;

    println!("healthy: {:?}", pick(Some(95.0)));
    println!("warning: {:?}", pick(Some(75.0)));
    println!("critical: {:?}", pick(Some(20.0)));
    println!("unknown: {:?}", pick(None));
    println!("invalid: {:?}", pick(Some(150.0)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn green_when_high() {
        let v = pick(Some(95.0));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 32);
        }
    }

    #[test]
    fn yellow_in_middle() {
        let v = pick(Some(75.0));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 33);
        }
    }

    #[test]
    fn red_when_low() {
        let v = pick(Some(20.0));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 31);
        }
    }

    #[test]
    fn boundary_at_90_green() {
        let v = pick(Some(90.0));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 32);
        }
    }

    #[test]
    fn boundary_at_50_yellow() {
        let v = pick(Some(50.0));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 33);
        }
    }

    #[test]
    fn just_below_50_red() {
        let v = pick(Some(49.9));
        if let DotVerdict::Pick { ansi_color, .. } = v {
            assert_eq!(ansi_color, 31);
        }
    }

    #[test]
    fn none_unknown() {
        assert_eq!(pick(None), DotVerdict::Unknown);
    }

    #[test]
    fn over_100_invalid() {
        assert_eq!(pick(Some(150.0)), DotVerdict::InvalidHealth);
    }

    #[test]
    fn negative_invalid() {
        assert_eq!(pick(Some(-10.0)), DotVerdict::InvalidHealth);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(pick(Some(f64::NAN)), DotVerdict::InvalidHealth);
    }

    #[test]
    fn deterministic() {
        let a = pick(Some(95.0));
        let b = pick(Some(95.0));
        assert_eq!(a, b);
    }
}
