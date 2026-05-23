//! # TUI Chart Axis Ticks
//!
//! Compute "nice" axis ticks for a (min, max) range and approximate
//! tick count. Returns the actual ticks, which use 1, 2, or 5 × 10^k
//! intervals (similar to matplotlib MaxNLocator).
//!
//! Demonstrates the **TUI.34** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: matplotlib MaxNLocator + Heckbert "nice numbers" (1990).
//!
//! Run with: cargo run --example tui_chart_axis_ticks
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TickVerdict {
    Ok { ticks: Vec<f64>, step: f64 },
    InvalidRange,
}

pub fn compute(min: f64, max: f64, target_ticks: u32) -> TickVerdict {
    if !min.is_finite() || !max.is_finite() || max <= min || target_ticks < 2 {
        return TickVerdict::InvalidRange;
    }
    let range = max - min;
    let raw_step = range / f64::from(target_ticks - 1);
    let exp = raw_step.log10().floor();
    let pow = 10.0_f64.powf(exp);
    let frac = raw_step / pow;
    let nice_frac = if frac < 1.5 {
        1.0
    } else if frac < 3.5 {
        2.0
    } else if frac < 7.5 {
        5.0
    } else {
        10.0
    };
    let step = nice_frac * pow;
    let start = (min / step).floor() * step;
    let mut ticks = Vec::new();
    let mut t = start;
    while t <= max + step / 2.0 {
        if t >= min - step / 2.0 {
            ticks.push(t);
        }
        t += step;
    }
    TickVerdict::Ok { ticks, step }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_chart_axis_ticks")?;

    println!("0..100, 5: {:?}", compute(0.0, 100.0, 5));
    println!("0..1, 5: {:?}", compute(0.0, 1.0, 5));
    println!("invalid: {:?}", compute(10.0, 5.0, 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nice_step_for_zero_to_100() {
        // raw step = 100/4 = 25 → frac = 2.5 → nice = 2 → step = 20.
        let v = compute(0.0, 100.0, 5);
        if let TickVerdict::Ok { step, .. } = v {
            assert!((step - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn ticks_cover_range() {
        let v = compute(0.0, 100.0, 5);
        if let TickVerdict::Ok { ticks, .. } = v {
            assert!(ticks.first().unwrap_or(&100.0) <= &0.0);
            assert!(ticks.last().unwrap_or(&-1.0) >= &100.0);
        }
    }

    #[test]
    fn small_range_works() {
        let v = compute(0.0, 1.0, 5);
        if let TickVerdict::Ok { step, .. } = v {
            assert!(step <= 0.5);
        }
    }

    #[test]
    fn invalid_min_ge_max() {
        assert_eq!(compute(10.0, 5.0, 5), TickVerdict::InvalidRange);
    }

    #[test]
    fn invalid_target_under_2() {
        assert_eq!(compute(0.0, 10.0, 1), TickVerdict::InvalidRange);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(compute(f64::NAN, 10.0, 5), TickVerdict::InvalidRange);
    }

    #[test]
    fn negative_range_works() {
        let v = compute(-10.0, 10.0, 5);
        assert!(matches!(v, TickVerdict::Ok { .. }));
    }

    #[test]
    fn step_positive() {
        let v = compute(0.0, 100.0, 5);
        if let TickVerdict::Ok { step, .. } = v {
            assert!(step > 0.0);
        }
    }

    #[test]
    fn ticks_increasing() {
        let v = compute(0.0, 100.0, 5);
        if let TickVerdict::Ok { ticks, .. } = v {
            for w in ticks.windows(2) {
                assert!(w[1] > w[0]);
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = compute(0.0, 100.0, 5);
        let b = compute(0.0, 100.0, 5);
        assert_eq!(a, b);
    }
}
