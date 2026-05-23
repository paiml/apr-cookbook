//! # TUI Search Jump List
//!
//! Given a sorted list of match positions and the current cursor,
//! return the next or previous match position with wrap-around.
//!
//! Demonstrates the **TUI.84** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim n/N forward-back search; less-style "Find Next".
//!
//! Run with: cargo run --example tui_search_jump_list
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Direction {
    Next,
    Previous,
}

#[derive(Debug, PartialEq)]
pub enum JumpVerdict {
    Ok { target: u32, wrapped: bool },
    NoMatches,
    InvalidConfig,
}

pub fn jump(matches: &[u32], cursor: u32, direction: Direction) -> JumpVerdict {
    if matches.is_empty() {
        return JumpVerdict::NoMatches;
    }
    // Verify sorted ascending.
    for w in matches.windows(2) {
        if w[0] >= w[1] {
            return JumpVerdict::InvalidConfig;
        }
    }
    match direction {
        Direction::Next => {
            for &m in matches {
                if m > cursor {
                    return JumpVerdict::Ok {
                        target: m,
                        wrapped: false,
                    };
                }
            }
            // Wrap to first.
            JumpVerdict::Ok {
                target: matches[0],
                wrapped: true,
            }
        }
        Direction::Previous => {
            for &m in matches.iter().rev() {
                if m < cursor {
                    return JumpVerdict::Ok {
                        target: m,
                        wrapped: false,
                    };
                }
            }
            // Wrap to last.
            JumpVerdict::Ok {
                target: matches[matches.len() - 1],
                wrapped: true,
            }
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_search_jump_list")?;

    let matches = [10, 25, 40, 75];
    println!("next from 30: {:?}", jump(&matches, 30, Direction::Next));
    println!(
        "prev from 30: {:?}",
        jump(&matches, 30, Direction::Previous)
    );
    println!(
        "wrap next from 80: {:?}",
        jump(&matches, 80, Direction::Next)
    );
    println!("no matches: {:?}", jump(&[], 0, Direction::Next));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jumper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn next_after_cursor() {
        let m = [10, 20, 30];
        let v = jump(&m, 15, Direction::Next);
        if let JumpVerdict::Ok { target, wrapped } = v {
            assert_eq!(target, 20);
            assert!(!wrapped);
        }
    }

    #[test]
    fn previous_before_cursor() {
        let m = [10, 20, 30];
        let v = jump(&m, 25, Direction::Previous);
        if let JumpVerdict::Ok { target, wrapped } = v {
            assert_eq!(target, 20);
            assert!(!wrapped);
        }
    }

    #[test]
    fn next_wraps_to_first() {
        let m = [10, 20, 30];
        let v = jump(&m, 100, Direction::Next);
        if let JumpVerdict::Ok { target, wrapped } = v {
            assert_eq!(target, 10);
            assert!(wrapped);
        }
    }

    #[test]
    fn previous_wraps_to_last() {
        let m = [10, 20, 30];
        let v = jump(&m, 0, Direction::Previous);
        if let JumpVerdict::Ok { target, wrapped } = v {
            assert_eq!(target, 30);
            assert!(wrapped);
        }
    }

    #[test]
    fn empty_matches_no_matches() {
        assert_eq!(jump(&[], 0, Direction::Next), JumpVerdict::NoMatches);
    }

    #[test]
    fn unsorted_rejected() {
        let m = [30, 10, 20];
        assert_eq!(jump(&m, 15, Direction::Next), JumpVerdict::InvalidConfig);
    }

    #[test]
    fn duplicate_rejected() {
        let m = [10, 10, 20];
        assert_eq!(jump(&m, 5, Direction::Next), JumpVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let m = [10, 20];
        let r1 = jump(&m, 15, Direction::Next);
        let r2 = jump(&m, 15, Direction::Next);
        assert_eq!(r1, r2);
    }

    #[test]
    fn cursor_at_match_skips_to_next() {
        let m = [10, 20, 30];
        let v = jump(&m, 20, Direction::Next);
        if let JumpVerdict::Ok { target, .. } = v {
            assert_eq!(target, 30);
        }
    }

    #[test]
    fn cursor_at_match_skips_to_previous() {
        let m = [10, 20, 30];
        let v = jump(&m, 20, Direction::Previous);
        if let JumpVerdict::Ok { target, .. } = v {
            assert_eq!(target, 10);
        }
    }

    #[test]
    fn single_match_wraps_both_directions() {
        let m = [50];
        let next = jump(&m, 100, Direction::Next);
        let prev = jump(&m, 0, Direction::Previous);
        if let (JumpVerdict::Ok { target: n, .. }, JumpVerdict::Ok { target: p, .. }) = (next, prev)
        {
            assert_eq!(n, 50);
            assert_eq!(p, 50);
        }
    }
}
