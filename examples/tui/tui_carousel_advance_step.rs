//! # TUI Carousel Advance Step
//!
//! Advance carousel position with optional wrap. Supports next, prev,
//! and "go to N" jumps. Returns next index and whether wrap occurred.
//!
//! Demonstrates the **TUI.161** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bootstrap carousel `data-bs-slide` semantics; Material
//!  Design carousel cycle behavior.
//!
//! Run with: cargo run --example tui_carousel_advance_step
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CarouselVerdict {
    Ok { next_idx: u32, wrapped: bool },
    InvalidConfig,
}

pub fn advance(slide_count: u32, current_idx: u32, direction: &str, wrap: bool) -> CarouselVerdict {
    if slide_count == 0 || current_idx >= slide_count {
        return CarouselVerdict::InvalidConfig;
    }
    let last = slide_count - 1;
    match direction {
        "next" => {
            if current_idx == last {
                if wrap {
                    CarouselVerdict::Ok {
                        next_idx: 0,
                        wrapped: true,
                    }
                } else {
                    CarouselVerdict::Ok {
                        next_idx: last,
                        wrapped: false,
                    }
                }
            } else {
                CarouselVerdict::Ok {
                    next_idx: current_idx + 1,
                    wrapped: false,
                }
            }
        }
        "prev" => {
            if current_idx == 0 {
                if wrap {
                    CarouselVerdict::Ok {
                        next_idx: last,
                        wrapped: true,
                    }
                } else {
                    CarouselVerdict::Ok {
                        next_idx: 0,
                        wrapped: false,
                    }
                }
            } else {
                CarouselVerdict::Ok {
                    next_idx: current_idx - 1,
                    wrapped: false,
                }
            }
        }
        _ => {
            // Try parsing as number for "goto"
            match direction.parse::<u32>() {
                Ok(n) if n < slide_count => CarouselVerdict::Ok {
                    next_idx: n,
                    wrapped: false,
                },
                _ => CarouselVerdict::InvalidConfig,
            }
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_carousel_advance_step")?;

    println!("next: {:?}", advance(3, 0, "next", false));
    println!("prev wrap: {:?}", advance(3, 0, "prev", true));
    println!("goto 2: {:?}", advance(3, 0, "2", false));
    println!("invalid: {:?}", advance(0, 0, "next", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advancer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_slides() {
        assert_eq!(advance(0, 0, "next", false), CarouselVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_idx_oob() {
        assert_eq!(advance(3, 5, "next", false), CarouselVerdict::InvalidConfig);
    }

    #[test]
    fn next_advances() {
        let v = advance(3, 0, "next", false);
        if let CarouselVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 1);
        }
    }

    #[test]
    fn next_at_end_no_wrap_stays() {
        let v = advance(3, 2, "next", false);
        if let CarouselVerdict::Ok { next_idx, wrapped } = v {
            assert_eq!(next_idx, 2);
            assert!(!wrapped);
        }
    }

    #[test]
    fn next_at_end_wraps_to_zero() {
        let v = advance(3, 2, "next", true);
        if let CarouselVerdict::Ok { next_idx, wrapped } = v {
            assert_eq!(next_idx, 0);
            assert!(wrapped);
        }
    }

    #[test]
    fn prev_decrements() {
        let v = advance(3, 1, "prev", false);
        if let CarouselVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 0);
        }
    }

    #[test]
    fn prev_at_zero_wraps_to_last() {
        let v = advance(3, 0, "prev", true);
        if let CarouselVerdict::Ok { next_idx, wrapped } = v {
            assert_eq!(next_idx, 2);
            assert!(wrapped);
        }
    }

    #[test]
    fn goto_valid_index() {
        let v = advance(5, 0, "3", false);
        if let CarouselVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 3);
        }
    }

    #[test]
    fn goto_oob_invalid() {
        assert_eq!(advance(3, 0, "5", false), CarouselVerdict::InvalidConfig);
    }

    #[test]
    fn unknown_direction_invalid() {
        assert_eq!(advance(3, 0, "wat", false), CarouselVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = advance(3, 0, "next", false);
        let r2 = advance(3, 0, "next", false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_slides_handled() {
        let v = advance(100, 50, "next", false);
        if let CarouselVerdict::Ok { next_idx, .. } = v {
            assert_eq!(next_idx, 51);
        }
    }
}
