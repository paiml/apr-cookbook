//! # TUI Animation Easing Curve
//!
//! Evaluate a discrete easing function at parameter `t` ∈ [0,1].
//! Supports Linear, EaseIn (quadratic), EaseOut (quadratic),
//! EaseInOut (cubic), and Bounce.
//!
//! Demonstrates the **TUI.87** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Robert Penner's easing equations (2001); CSS Animations
//!  cubic-bezier function.
//!
//! Run with: cargo run --example tui_animation_easing_curve
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Easing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
}

#[derive(Debug, PartialEq)]
pub enum EasingVerdict {
    Ok { value: f64 },
    InvalidConfig,
}

pub fn evaluate(easing: Easing, t: f64) -> EasingVerdict {
    if !(0.0..=1.0).contains(&t) {
        return EasingVerdict::InvalidConfig;
    }
    let v = match easing {
        Easing::Linear => t,
        Easing::EaseIn => t * t,
        Easing::EaseOut => 1.0 - (1.0 - t).powi(2),
        Easing::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
            }
        }
        Easing::Bounce => {
            // Simple "snap back" near end.
            let phase = (t * std::f64::consts::PI).sin();
            t * (1.0 + 0.1 * phase)
        }
    };
    EasingVerdict::Ok { value: v }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_animation_easing_curve")?;

    println!("linear: {:?}", evaluate(Easing::Linear, 0.5));
    println!("ease-in: {:?}", evaluate(Easing::EaseIn, 0.5));
    println!("ease-out: {:?}", evaluate(Easing::EaseOut, 0.5));
    println!("ease-in-out: {:?}", evaluate(Easing::EaseInOut, 0.5));
    println!("invalid: {:?}", evaluate(Easing::Linear, -0.1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_t_equals_value() {
        let v = evaluate(Easing::Linear, 0.5);
        if let EasingVerdict::Ok { value } = v {
            assert!((value - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn ease_in_starts_slow() {
        let v = evaluate(Easing::EaseIn, 0.25);
        if let EasingVerdict::Ok { value } = v {
            // 0.25^2 = 0.0625
            assert!(value < 0.25);
        }
    }

    #[test]
    fn ease_out_starts_fast() {
        let v = evaluate(Easing::EaseOut, 0.25);
        if let EasingVerdict::Ok { value } = v {
            assert!(value > 0.25);
        }
    }

    #[test]
    fn easings_meet_at_endpoints() {
        for &easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
        ]
        .iter()
        {
            if let EasingVerdict::Ok { value } = evaluate(easing, 0.0) {
                assert!(value.abs() < 1e-9);
            }
            if let EasingVerdict::Ok { value } = evaluate(easing, 1.0) {
                assert!((value - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ease_in_out_passes_half_at_midpoint() {
        let v = evaluate(Easing::EaseInOut, 0.5);
        if let EasingVerdict::Ok { value } = v {
            assert!((value - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_t_negative() {
        assert_eq!(evaluate(Easing::Linear, -0.1), EasingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_t_above_one() {
        assert_eq!(evaluate(Easing::Linear, 1.5), EasingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = evaluate(Easing::EaseIn, 0.3);
        let r2 = evaluate(Easing::EaseIn, 0.3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn linear_monotonic() {
        let lo = evaluate(Easing::Linear, 0.2);
        let hi = evaluate(Easing::Linear, 0.8);
        if let (EasingVerdict::Ok { value: l }, EasingVerdict::Ok { value: h }) = (lo, hi) {
            assert!(h > l);
        }
    }

    #[test]
    fn bounce_value_in_unit_range_or_close() {
        let v = evaluate(Easing::Bounce, 0.5);
        if let EasingVerdict::Ok { value } = v {
            assert!((-0.1..=1.2).contains(&value));
        }
    }

    #[test]
    fn ease_in_out_symmetric() {
        let lo = evaluate(Easing::EaseInOut, 0.25);
        let hi = evaluate(Easing::EaseInOut, 0.75);
        if let (EasingVerdict::Ok { value: l }, EasingVerdict::Ok { value: h }) = (lo, hi) {
            // f(0.25) + f(0.75) ≈ 1.0 (symmetry).
            assert!((l + h - 1.0).abs() < 1e-9);
        }
    }
}
