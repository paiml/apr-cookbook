//! # TUI Gradient Fill Compute
//!
//! Compute interpolated RGB values for a linear gradient between two
//! colors over N steps. Returns the gradient strip as `(r,g,b)` tuples.
//!
//! Demonstrates the **TUI.150** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS `linear-gradient` interpolation; ANSI 24-bit color
//!  escape sequences (xterm 256-color tables).
//!
//! Run with: cargo run --example tui_gradient_fill_compute
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GradientVerdict {
    Ok {
        strip: Vec<(u8, u8, u8)>,
        steps: u32,
    },
    InvalidConfig,
}

pub fn compute(start: (u8, u8, u8), end: (u8, u8, u8), steps: u32) -> GradientVerdict {
    if steps < 2 {
        return GradientVerdict::InvalidConfig;
    }
    let mut strip: Vec<(u8, u8, u8)> = Vec::with_capacity(steps as usize);
    let denom = (steps - 1) as f64;
    for i in 0..steps {
        let t = i as f64 / denom;
        let r = (start.0 as f64 + t * (end.0 as f64 - start.0 as f64)) as u8;
        let g = (start.1 as f64 + t * (end.1 as f64 - start.1 as f64)) as u8;
        let b = (start.2 as f64 + t * (end.2 as f64 - start.2 as f64)) as u8;
        strip.push((r, g, b));
    }
    GradientVerdict::Ok { strip, steps }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_gradient_fill_compute")?;

    println!(
        "black-to-white: {:?}",
        compute((0, 0, 0), (255, 255, 255), 5)
    );
    println!("invalid: {:?}", compute((0, 0, 0), (255, 255, 255), 1));
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
    fn invalid_one_step() {
        assert_eq!(
            compute((0, 0, 0), (255, 0, 0), 1),
            GradientVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            compute((0, 0, 0), (255, 0, 0), 0),
            GradientVerdict::InvalidConfig
        );
    }

    #[test]
    fn endpoints_match() {
        let v = compute((10, 20, 30), (100, 200, 250), 5);
        if let GradientVerdict::Ok { strip, .. } = v {
            assert_eq!(strip.first(), Some(&(10, 20, 30)));
            assert_eq!(strip.last(), Some(&(100, 200, 250)));
        }
    }

    #[test]
    fn step_count_matches() {
        let v = compute((0, 0, 0), (255, 255, 255), 10);
        if let GradientVerdict::Ok { strip, .. } = v {
            assert_eq!(strip.len(), 10);
        }
    }

    #[test]
    fn two_steps_just_endpoints() {
        let v = compute((0, 0, 0), (255, 255, 255), 2);
        if let GradientVerdict::Ok { strip, .. } = v {
            assert_eq!(strip, vec![(0, 0, 0), (255, 255, 255)]);
        }
    }

    #[test]
    fn middle_is_average_for_three() {
        let v = compute((0, 0, 0), (200, 200, 200), 3);
        if let GradientVerdict::Ok { strip, .. } = v {
            assert_eq!(strip[1], (100, 100, 100));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute((0, 0, 0), (255, 255, 255), 5);
        let r2 = compute((0, 0, 0), (255, 255, 255), 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn monotone_red_channel_when_increasing() {
        let v = compute((0, 0, 0), (255, 0, 0), 10);
        if let GradientVerdict::Ok { strip, .. } = v {
            for w in strip.windows(2) {
                assert!(w[0].0 <= w[1].0);
            }
        }
    }

    #[test]
    fn monotone_red_channel_when_decreasing() {
        let v = compute((255, 0, 0), (0, 0, 0), 10);
        if let GradientVerdict::Ok { strip, .. } = v {
            for w in strip.windows(2) {
                assert!(w[0].0 >= w[1].0);
            }
        }
    }

    #[test]
    fn many_steps_handled() {
        let v = compute((0, 0, 0), (255, 255, 255), 100);
        if let GradientVerdict::Ok { strip, .. } = v {
            assert_eq!(strip.len(), 100);
        }
    }

    #[test]
    fn same_endpoint_constant() {
        let v = compute((100, 100, 100), (100, 100, 100), 5);
        if let GradientVerdict::Ok { strip, .. } = v {
            for c in &strip {
                assert_eq!(*c, (100, 100, 100));
            }
        }
    }

    #[test]
    fn steps_returned() {
        let v = compute((0, 0, 0), (255, 255, 255), 7);
        if let GradientVerdict::Ok { steps, .. } = v {
            assert_eq!(steps, 7);
        }
    }
}
