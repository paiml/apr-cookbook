//! # TUI Gauge Meter Render
//!
//! Render a horizontal gauge meter showing current value against a
//! max range. Returns the meter string with filled/empty proportion
//! and the percentage label.
//!
//! Demonstrates the **TUI.165** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: htop CPU/memory bar rendering; HUD-style game UI gauges.
//!
//! Run with: cargo run --example tui_gauge_meter_render
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GaugeVerdict {
    Ok { rendered: String, percent: u32 },
    InvalidConfig,
}

pub fn render(value: u32, max: u32, width: u32) -> GaugeVerdict {
    if max == 0 || value > max || !(5..=100).contains(&width) {
        return GaugeVerdict::InvalidConfig;
    }
    let pct = (value as u64 * 100 / max as u64) as u32;
    let filled = (value as u64 * width as u64 / max as u64) as u32;
    let empty = width - filled;
    let bar = format!(
        "[{}{}] {pct}%",
        "█".repeat(filled as usize),
        "·".repeat(empty as usize),
        pct = pct,
    );
    GaugeVerdict::Ok {
        rendered: bar,
        percent: pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_gauge_meter_render")?;

    println!("50%: {:?}", render(50, 100, 20));
    println!("100%: {:?}", render(100, 100, 20));
    println!("0%: {:?}", render(0, 100, 20));
    println!("invalid: {:?}", render(150, 100, 20));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(render(50, 0, 20), GaugeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_value_over_max() {
        assert_eq!(render(150, 100, 20), GaugeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_width_too_small() {
        assert_eq!(render(50, 100, 2), GaugeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_width_too_large() {
        assert_eq!(render(50, 100, 200), GaugeVerdict::InvalidConfig);
    }

    #[test]
    fn fifty_percent_correct() {
        let v = render(50, 100, 20);
        if let GaugeVerdict::Ok { percent, .. } = v {
            assert_eq!(percent, 50);
        }
    }

    #[test]
    fn full_value_full_bar() {
        let v = render(100, 100, 20);
        if let GaugeVerdict::Ok { rendered, .. } = v {
            let filled = rendered.matches('█').count();
            assert_eq!(filled, 20);
        }
    }

    #[test]
    fn zero_value_empty_bar() {
        let v = render(0, 100, 20);
        if let GaugeVerdict::Ok { rendered, .. } = v {
            assert!(!rendered.contains('█'));
        }
    }

    #[test]
    fn brackets_present() {
        let v = render(50, 100, 20);
        if let GaugeVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('['));
            assert!(rendered.contains(']'));
        }
    }

    #[test]
    fn percent_in_output() {
        let v = render(75, 100, 20);
        if let GaugeVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("75%"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(50, 100, 20);
        let r2 = render(50, 100, 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_value_equals_max() {
        let v = render(100, 100, 20);
        if let GaugeVerdict::Ok { percent, .. } = v {
            assert_eq!(percent, 100);
        }
    }

    #[test]
    fn min_width_accepted() {
        let v = render(50, 100, 5);
        assert!(matches!(v, GaugeVerdict::Ok { .. }));
    }

    #[test]
    fn max_width_accepted() {
        let v = render(50, 100, 100);
        assert!(matches!(v, GaugeVerdict::Ok { .. }));
    }

    #[test]
    fn percent_truncates_not_rounds() {
        // 33/100 = 33% (integer truncation, not 33.33→33).
        let v = render(33, 100, 20);
        if let GaugeVerdict::Ok { percent, .. } = v {
            assert_eq!(percent, 33);
        }
    }
}
