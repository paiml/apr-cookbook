//! # TUI Resize Constraints
//!
//! Apply min/max constraints to a resize event. If requested size is
//! within bounds, accept; else clamp and report what was clamped.
//!
//! Demonstrates the **TUI.24** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK widget min/max-width constraints.
//!
//! Run with: cargo run --example tui_resize_constraints
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ResizeVerdict {
    Accepted {
        new_width: u32,
        new_height: u32,
    },
    Clamped {
        new_width: u32,
        new_height: u32,
        clamped: &'static str,
    },
    InvalidConstraints,
}

pub fn apply(
    requested_width: u32,
    requested_height: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
) -> ResizeVerdict {
    if min_width > max_width || min_height > max_height || max_width == 0 || max_height == 0 {
        return ResizeVerdict::InvalidConstraints;
    }
    let new_width = requested_width.clamp(min_width, max_width);
    let new_height = requested_height.clamp(min_height, max_height);
    let w_clamped = new_width != requested_width;
    let h_clamped = new_height != requested_height;
    let clamped = match (w_clamped, h_clamped) {
        (true, true) => "both",
        (true, false) => "width",
        (false, true) => "height",
        (false, false) => "",
    };
    if clamped.is_empty() {
        ResizeVerdict::Accepted {
            new_width,
            new_height,
        }
    } else {
        ResizeVerdict::Clamped {
            new_width,
            new_height,
            clamped,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_resize_constraints")?;

    println!("accepted: {:?}", apply(80, 24, 40, 200, 10, 100));
    println!("width clamped: {:?}", apply(300, 24, 40, 200, 10, 100));
    println!("both clamped: {:?}", apply(300, 200, 40, 200, 10, 100));
    println!("invalid: {:?}", apply(80, 24, 200, 100, 10, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_bounds_accepted() {
        let v = apply(80, 24, 40, 200, 10, 100);
        assert!(matches!(v, ResizeVerdict::Accepted { .. }));
    }

    #[test]
    fn over_max_width_clamped() {
        let v = apply(300, 24, 40, 200, 10, 100);
        if let ResizeVerdict::Clamped {
            new_width, clamped, ..
        } = v
        {
            assert_eq!(new_width, 200);
            assert_eq!(clamped, "width");
        }
    }

    #[test]
    fn under_min_width_clamped() {
        let v = apply(20, 24, 40, 200, 10, 100);
        if let ResizeVerdict::Clamped { new_width, .. } = v {
            assert_eq!(new_width, 40);
        }
    }

    #[test]
    fn over_max_height_clamped() {
        let v = apply(80, 200, 40, 200, 10, 100);
        if let ResizeVerdict::Clamped {
            new_height,
            clamped,
            ..
        } = v
        {
            assert_eq!(new_height, 100);
            assert_eq!(clamped, "height");
        }
    }

    #[test]
    fn both_clamped() {
        let v = apply(300, 200, 40, 200, 10, 100);
        if let ResizeVerdict::Clamped { clamped, .. } = v {
            assert_eq!(clamped, "both");
        }
    }

    #[test]
    fn invalid_min_over_max() {
        assert_eq!(
            apply(80, 24, 200, 100, 10, 100),
            ResizeVerdict::InvalidConstraints
        );
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(
            apply(80, 24, 0, 0, 10, 100),
            ResizeVerdict::InvalidConstraints
        );
    }

    #[test]
    fn at_min_accepted() {
        let v = apply(40, 10, 40, 200, 10, 100);
        assert!(matches!(v, ResizeVerdict::Accepted { .. }));
    }

    #[test]
    fn at_max_accepted() {
        let v = apply(200, 100, 40, 200, 10, 100);
        assert!(matches!(v, ResizeVerdict::Accepted { .. }));
    }

    #[test]
    fn deterministic() {
        let a = apply(80, 24, 40, 200, 10, 100);
        let b = apply(80, 24, 40, 200, 10, 100);
        assert_eq!(a, b);
    }
}
