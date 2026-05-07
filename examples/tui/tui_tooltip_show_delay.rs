//! # TUI Tooltip Show Delay
//!
//! Compute tooltip visibility based on hover time: shows after
//! `show_delay_ms`, hides after `hide_delay_ms` once cursor leaves.
//! Returns `Show`, `Hide`, or `Pending` verdict + ms-since-event.
//!
//! Demonstrates the **TUI.141** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS NSToolTip default delay (1.5s); GTK
//!  `gtk-tooltip-timeout` default (500 ms).
//!
//! Run with: cargo run --example tui_tooltip_show_delay
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TooltipVerdict {
    Show { visible_for_ms: u32 },
    Hide { hidden_for_ms: u32 },
    Pending { ms_remaining: u32 },
    InvalidConfig,
}

pub fn evaluate(
    elapsed_ms: u32,
    is_hovering: bool,
    show_delay_ms: u32,
    hide_delay_ms: u32,
) -> TooltipVerdict {
    if show_delay_ms == 0 || hide_delay_ms == 0 {
        return TooltipVerdict::InvalidConfig;
    }
    if is_hovering {
        if elapsed_ms >= show_delay_ms {
            TooltipVerdict::Show {
                visible_for_ms: elapsed_ms - show_delay_ms,
            }
        } else {
            TooltipVerdict::Pending {
                ms_remaining: show_delay_ms - elapsed_ms,
            }
        }
    } else if elapsed_ms >= hide_delay_ms {
        TooltipVerdict::Hide {
            hidden_for_ms: elapsed_ms - hide_delay_ms,
        }
    } else {
        TooltipVerdict::Pending {
            ms_remaining: hide_delay_ms - elapsed_ms,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_tooltip_show_delay")?;

    println!("hovering early: {:?}", evaluate(100, true, 500, 200));
    println!("hovering long: {:?}", evaluate(700, true, 500, 200));
    println!("just left: {:?}", evaluate(50, false, 500, 200));
    println!("invalid: {:?}", evaluate(0, true, 0, 0));
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
    fn hover_pending_when_under_delay() {
        let v = evaluate(100, true, 500, 200);
        assert_eq!(v, TooltipVerdict::Pending { ms_remaining: 400 });
    }

    #[test]
    fn hover_show_after_delay() {
        let v = evaluate(700, true, 500, 200);
        assert_eq!(
            v,
            TooltipVerdict::Show {
                visible_for_ms: 200
            }
        );
    }

    #[test]
    fn leave_pending_when_under_delay() {
        let v = evaluate(50, false, 500, 200);
        assert_eq!(v, TooltipVerdict::Pending { ms_remaining: 150 });
    }

    #[test]
    fn leave_hide_after_delay() {
        let v = evaluate(300, false, 500, 200);
        assert_eq!(v, TooltipVerdict::Hide { hidden_for_ms: 100 });
    }

    #[test]
    fn invalid_zero_show_delay() {
        assert_eq!(evaluate(0, true, 0, 200), TooltipVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_hide_delay() {
        assert_eq!(evaluate(0, false, 500, 0), TooltipVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_show_at_delay() {
        // elapsed == show_delay → just shown.
        let v = evaluate(500, true, 500, 200);
        assert_eq!(v, TooltipVerdict::Show { visible_for_ms: 0 });
    }

    #[test]
    fn boundary_hide_at_delay() {
        let v = evaluate(200, false, 500, 200);
        assert_eq!(v, TooltipVerdict::Hide { hidden_for_ms: 0 });
    }

    #[test]
    fn deterministic() {
        let r1 = evaluate(100, true, 500, 200);
        let r2 = evaluate(100, true, 500, 200);
        assert_eq!(r1, r2);
    }

    #[test]
    fn long_visible_duration() {
        let v = evaluate(2_000_000, true, 500, 200);
        assert_eq!(
            v,
            TooltipVerdict::Show {
                visible_for_ms: 1_999_500
            }
        );
    }

    #[test]
    fn pending_remaining_decreases_with_elapsed() {
        // At elapsed=100 vs 200, ms_remaining should decrease.
        let early = evaluate(100, true, 500, 200);
        let later = evaluate(200, true, 500, 200);
        if let (
            TooltipVerdict::Pending { ms_remaining: a },
            TooltipVerdict::Pending { ms_remaining: b },
        ) = (early, later)
        {
            assert!(b < a);
        }
    }

    #[test]
    fn hover_state_independent_outcome() {
        // Same elapsed, different hover → different state.
        let h = evaluate(1000, true, 500, 200);
        let n = evaluate(1000, false, 500, 200);
        assert!(h != n);
    }
}
