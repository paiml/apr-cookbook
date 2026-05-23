//! # TUI Split Pane Resize
//!
//! Resize a vertical split-pane by dragging the divider. Returns the
//! new (left, right) widths, clamping to min sizes for each pane.
//!
//! Demonstrates the **TUI.41** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: tmux split-pane resize / GIMP dock dividers.
//!
//! Run with: cargo run --example tui_split_pane_resize
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ResizeVerdict {
    Ok {
        left: u32,
        right: u32,
    },
    Clamped {
        left: u32,
        right: u32,
        side: &'static str,
    },
    InvalidConfig,
}

pub fn resize(
    total_width: u32,
    requested_left: u32,
    min_left: u32,
    min_right: u32,
) -> ResizeVerdict {
    if total_width == 0 || min_left + min_right > total_width {
        return ResizeVerdict::InvalidConfig;
    }
    let max_left = total_width - min_right;
    if requested_left < min_left {
        let right = total_width - min_left;
        return ResizeVerdict::Clamped {
            left: min_left,
            right,
            side: "left",
        };
    }
    if requested_left > max_left {
        let right = total_width - max_left;
        return ResizeVerdict::Clamped {
            left: max_left,
            right,
            side: "right",
        };
    }
    ResizeVerdict::Ok {
        left: requested_left,
        right: total_width - requested_left,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_split_pane_resize")?;

    println!("ok: {:?}", resize(100, 60, 20, 20));
    println!("clamp left: {:?}", resize(100, 5, 20, 20));
    println!("clamp right: {:?}", resize(100, 90, 20, 20));
    println!("invalid: {:?}", resize(100, 50, 60, 60));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_bounds_ok() {
        let v = resize(100, 60, 20, 20);
        if let ResizeVerdict::Ok { left, right } = v {
            assert_eq!(left, 60);
            assert_eq!(right, 40);
        }
    }

    #[test]
    fn clamp_left_min() {
        let v = resize(100, 5, 20, 20);
        if let ResizeVerdict::Clamped { left, right, side } = v {
            assert_eq!(left, 20);
            assert_eq!(right, 80);
            assert_eq!(side, "left");
        }
    }

    #[test]
    fn clamp_right_min() {
        let v = resize(100, 90, 20, 20);
        if let ResizeVerdict::Clamped { left, right, side } = v {
            assert_eq!(left, 80);
            assert_eq!(right, 20);
            assert_eq!(side, "right");
        }
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(resize(0, 50, 10, 10), ResizeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_min_too_large() {
        assert_eq!(resize(100, 50, 60, 60), ResizeVerdict::InvalidConfig);
    }

    #[test]
    fn min_equals_total_works() {
        let v = resize(40, 20, 20, 20);
        if let ResizeVerdict::Ok { left, right } = v {
            assert_eq!(left, 20);
            assert_eq!(right, 20);
        }
    }

    #[test]
    fn at_min_left_ok() {
        let v = resize(100, 20, 20, 20);
        assert!(matches!(v, ResizeVerdict::Ok { .. }));
    }

    #[test]
    fn at_max_left_ok() {
        let v = resize(100, 80, 20, 20);
        assert!(matches!(v, ResizeVerdict::Ok { .. }));
    }

    #[test]
    fn widths_sum_to_total() {
        for req in [5, 50, 95] {
            let v = resize(100, req, 20, 20);
            let (l, r) = match v {
                ResizeVerdict::Ok { left, right } => (left, right),
                ResizeVerdict::Clamped { left, right, .. } => (left, right),
                _ => panic!(),
            };
            assert_eq!(l + r, 100);
        }
    }

    #[test]
    fn deterministic() {
        let a = resize(100, 60, 20, 20);
        let b = resize(100, 60, 20, 20);
        assert_eq!(a, b);
    }
}
