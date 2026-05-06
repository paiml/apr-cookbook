//! # TUI Input Max-Length Indicator
//!
//! Render a counter `42/100` for input fields with max length, plus
//! warning state when within `warn_threshold` of limit.
//!
//! Demonstrates the **TUI.110** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 `<input maxlength>` UI; Twitter character counter.
//!
//! Run with: cargo run --example tui_input_max_length_indicator
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum CounterState {
    Ok,
    Warning,
    AtLimit,
    OverLimit,
}

#[derive(Debug, PartialEq)]
pub enum CounterVerdict {
    Ok {
        rendered: String,
        state: CounterState,
    },
    InvalidConfig,
}

pub fn render(current_len: u32, max_len: u32, warn_threshold: u32) -> CounterVerdict {
    if max_len == 0 {
        return CounterVerdict::InvalidConfig;
    }
    let rendered = format!("{current_len}/{max_len}");
    let state = if current_len > max_len {
        CounterState::OverLimit
    } else if current_len == max_len {
        CounterState::AtLimit
    } else if current_len + warn_threshold >= max_len {
        CounterState::Warning
    } else {
        CounterState::Ok
    };
    CounterVerdict::Ok { rendered, state }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_input_max_length_indicator")?;

    println!("ok: {:?}", render(50, 100, 10));
    println!("warning: {:?}", render(95, 100, 10));
    println!("at limit: {:?}", render(100, 100, 10));
    println!("over: {:?}", render(105, 100, 10));
    println!("invalid: {:?}", render(50, 0, 10));
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
    fn ok_state_below_warning() {
        let v = render(50, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::Ok);
        }
    }

    #[test]
    fn warning_state_in_threshold() {
        let v = render(95, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::Warning);
        }
    }

    #[test]
    fn at_limit_state() {
        let v = render(100, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::AtLimit);
        }
    }

    #[test]
    fn over_limit_state() {
        let v = render(105, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::OverLimit);
        }
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(render(50, 0, 10), CounterVerdict::InvalidConfig);
    }

    #[test]
    fn rendered_format_correct() {
        let v = render(42, 100, 10);
        if let CounterVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "42/100");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(50, 100, 10);
        let r2 = render(50, 100, 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_one_below_warning_ok() {
        let v = render(89, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::Ok);
        }
    }

    #[test]
    fn boundary_at_warning_warns() {
        let v = render(90, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::Warning);
        }
    }

    #[test]
    fn one_over_limit_over() {
        let v = render(101, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::OverLimit);
        }
    }

    #[test]
    fn empty_input_ok() {
        let v = render(0, 100, 10);
        if let CounterVerdict::Ok { state, .. } = v {
            assert_eq!(state, CounterState::Ok);
        }
    }

    #[test]
    fn small_max_length_works() {
        let v = render(5, 10, 2);
        if let CounterVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "5/10");
        }
    }
}
