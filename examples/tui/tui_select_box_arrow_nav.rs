//! # TUI Select Box Arrow Navigation
//!
//! Compute next-focus index for a select box on Up/Down keys with
//! optional wrap-around. Returns next index and whether the focus
//! wrapped.
//!
//! Demonstrates the **TUI.147** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML `<select>` arrow behavior (no wrap by default);
//!  vim wildmenu wrap-around mode.
//!
//! Run with: cargo run --example tui_select_box_arrow_nav
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SelectVerdict {
    Ok { next_idx: u32, wrapped: bool },
    InvalidConfig,
}

pub fn navigate(option_count: u32, current_idx: u32, key: &str, wrap: bool) -> SelectVerdict {
    if option_count == 0 || current_idx >= option_count {
        return SelectVerdict::InvalidConfig;
    }
    let last = option_count - 1;
    match key {
        "Down" => {
            if current_idx == last {
                if wrap {
                    SelectVerdict::Ok {
                        next_idx: 0,
                        wrapped: true,
                    }
                } else {
                    SelectVerdict::Ok {
                        next_idx: last,
                        wrapped: false,
                    }
                }
            } else {
                SelectVerdict::Ok {
                    next_idx: current_idx + 1,
                    wrapped: false,
                }
            }
        }
        "Up" => {
            if current_idx == 0 {
                if wrap {
                    SelectVerdict::Ok {
                        next_idx: last,
                        wrapped: true,
                    }
                } else {
                    SelectVerdict::Ok {
                        next_idx: 0,
                        wrapped: false,
                    }
                }
            } else {
                SelectVerdict::Ok {
                    next_idx: current_idx - 1,
                    wrapped: false,
                }
            }
        }
        "Home" => SelectVerdict::Ok {
            next_idx: 0,
            wrapped: false,
        },
        "End" => SelectVerdict::Ok {
            next_idx: last,
            wrapped: false,
        },
        _ => SelectVerdict::Ok {
            next_idx: current_idx,
            wrapped: false,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_select_box_arrow_nav")?;

    println!("down: {:?}", navigate(3, 0, "Down", false));
    println!("up at top, no wrap: {:?}", navigate(3, 0, "Up", false));
    println!("up at top, wrap: {:?}", navigate(3, 0, "Up", true));
    println!("invalid: {:?}", navigate(0, 0, "Down", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_options() {
        assert_eq!(navigate(0, 0, "Down", false), SelectVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_idx_oob() {
        assert_eq!(navigate(3, 5, "Down", false), SelectVerdict::InvalidConfig);
    }

    #[test]
    fn down_advances() {
        let v = navigate(3, 0, "Down", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 1);
        }
    }

    #[test]
    fn up_decrements() {
        let v = navigate(3, 1, "Up", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 0);
        }
    }

    #[test]
    fn down_at_end_no_wrap_stays() {
        let v = navigate(3, 2, "Down", false);
        if let SelectVerdict::Ok { next_idx, wrapped } = v {
            assert_eq!(next_idx, 2);
            assert!(!wrapped);
        }
    }

    #[test]
    fn down_at_end_wrap_to_zero() {
        let v = navigate(3, 2, "Down", true);
        assert_eq!(
            v,
            SelectVerdict::Ok {
                next_idx: 0,
                wrapped: true,
            }
        );
    }

    #[test]
    fn up_at_top_no_wrap_stays() {
        let v = navigate(3, 0, "Up", false);
        if let SelectVerdict::Ok { next_idx, wrapped } = v {
            assert_eq!(next_idx, 0);
            assert!(!wrapped);
        }
    }

    #[test]
    fn up_at_top_wrap_to_last() {
        let v = navigate(3, 0, "Up", true);
        assert_eq!(
            v,
            SelectVerdict::Ok {
                next_idx: 2,
                wrapped: true,
            }
        );
    }

    #[test]
    fn home_jumps_to_zero() {
        let v = navigate(3, 2, "Home", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 0);
        }
    }

    #[test]
    fn end_jumps_to_last() {
        let v = navigate(3, 0, "End", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = navigate(3, 0, "Down", false);
        let r2 = navigate(3, 0, "Down", false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unknown_key_keeps_idx() {
        let v = navigate(3, 1, "X", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 1);
        }
    }

    #[test]
    fn many_options_handled() {
        let v = navigate(100, 50, "Down", false);
        if let SelectVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 51);
        }
    }
}
