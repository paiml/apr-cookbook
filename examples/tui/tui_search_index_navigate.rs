//! # TUI Search Result Navigation
//!
//! Navigate among matches: Next / Previous / First / Last with
//! optional wrap-around. Returns the new match index.
//!
//! Demonstrates the **TUI.46** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: editor search navigation (Vim n/N, VS Code F3/Shift-F3).
//!
//! Run with: cargo run --example tui_search_index_navigate
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOp {
    Next,
    Previous,
    First,
    Last,
}

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok { index: u32 },
    NoMatches,
    InvalidConfig,
}

pub fn navigate(current: u32, match_count: u32, op: SearchOp, wrap: bool) -> SearchVerdict {
    if match_count == 0 {
        return SearchVerdict::NoMatches;
    }
    let cur = current.min(match_count - 1);
    let new_idx = match op {
        SearchOp::Next => {
            if cur + 1 >= match_count {
                if wrap {
                    0
                } else {
                    match_count - 1
                }
            } else {
                cur + 1
            }
        }
        SearchOp::Previous => {
            if cur == 0 {
                if wrap {
                    match_count - 1
                } else {
                    0
                }
            } else {
                cur - 1
            }
        }
        SearchOp::First => 0,
        SearchOp::Last => match_count - 1,
    };
    SearchVerdict::Ok { index: new_idx }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_search_index_navigate")?;

    println!("next: {:?}", navigate(0, 5, SearchOp::Next, false));
    println!(
        "next at end no wrap: {:?}",
        navigate(4, 5, SearchOp::Next, false)
    );
    println!(
        "next at end wrap: {:?}",
        navigate(4, 5, SearchOp::Next, true)
    );
    println!("first: {:?}", navigate(3, 5, SearchOp::First, false));
    println!("last: {:?}", navigate(0, 5, SearchOp::Last, false));
    println!("no matches: {:?}", navigate(0, 0, SearchOp::Next, false));
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
    fn next_advances() {
        let v = navigate(0, 5, SearchOp::Next, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn previous_decrements() {
        let v = navigate(2, 5, SearchOp::Previous, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn next_at_end_clamps() {
        let v = navigate(4, 5, SearchOp::Next, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn next_at_end_wraps() {
        let v = navigate(4, 5, SearchOp::Next, true);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn previous_at_start_clamps() {
        let v = navigate(0, 5, SearchOp::Previous, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn previous_at_start_wraps() {
        let v = navigate(0, 5, SearchOp::Previous, true);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn first_jumps_to_zero() {
        let v = navigate(3, 5, SearchOp::First, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn last_jumps_to_max() {
        let v = navigate(0, 5, SearchOp::Last, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn no_matches_special() {
        assert_eq!(
            navigate(0, 0, SearchOp::Next, false),
            SearchVerdict::NoMatches
        );
    }

    #[test]
    fn out_of_bounds_current_clamps() {
        let v = navigate(100, 5, SearchOp::Previous, false);
        if let SearchVerdict::Ok { index } = v {
            assert_eq!(index, 3);
        }
    }

    #[test]
    fn deterministic() {
        let a = navigate(2, 5, SearchOp::Next, false);
        let b = navigate(2, 5, SearchOp::Next, false);
        assert_eq!(a, b);
    }
}
