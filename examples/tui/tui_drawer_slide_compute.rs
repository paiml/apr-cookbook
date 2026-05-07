//! # TUI Drawer Slide Compute
//!
//! Compute drawer offset during a slide-in/slide-out animation given
//! elapsed time, total duration, and direction. Uses ease-out
//! interpolation. Returns offset and animation phase.
//!
//! Demonstrates the **TUI.156** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS `transition-timing-function: ease-out`; Material
//!  Design slide-in motion guidelines.
//!
//! Run with: cargo run --example tui_drawer_slide_compute
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SlidePhase {
    Hidden,
    Animating,
    FullyShown,
}

#[derive(Debug, PartialEq)]
pub enum SlideVerdict {
    Ok { offset: u32, phase: SlidePhase },
    InvalidConfig,
}

pub fn compute(
    elapsed_ms: u32,
    duration_ms: u32,
    drawer_width: u32,
    is_opening: bool,
) -> SlideVerdict {
    if duration_ms == 0 || drawer_width == 0 {
        return SlideVerdict::InvalidConfig;
    }
    if elapsed_ms == 0 {
        return SlideVerdict::Ok {
            offset: if is_opening { drawer_width } else { 0 },
            phase: if is_opening {
                SlidePhase::Hidden
            } else {
                SlidePhase::FullyShown
            },
        };
    }
    if elapsed_ms >= duration_ms {
        return SlideVerdict::Ok {
            offset: if is_opening { 0 } else { drawer_width },
            phase: if is_opening {
                SlidePhase::FullyShown
            } else {
                SlidePhase::Hidden
            },
        };
    }
    let t = elapsed_ms as f64 / duration_ms as f64;
    // Ease-out: 1 - (1-t)^3
    let eased = 1.0 - (1.0 - t).powi(3);
    let progress = if is_opening { 1.0 - eased } else { eased };
    let offset = (drawer_width as f64 * progress) as u32;
    SlideVerdict::Ok {
        offset,
        phase: SlidePhase::Animating,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_drawer_slide_compute")?;

    println!("opening 0ms: {:?}", compute(0, 300, 200, true));
    println!("opening 150ms: {:?}", compute(150, 300, 200, true));
    println!("opening 300ms: {:?}", compute(300, 300, 200, true));
    println!("invalid: {:?}", compute(0, 0, 200, true));
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
    fn invalid_zero_duration() {
        assert_eq!(compute(0, 0, 200, true), SlideVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(compute(0, 300, 0, true), SlideVerdict::InvalidConfig);
    }

    #[test]
    fn open_at_start_offset_full() {
        let v = compute(0, 300, 200, true);
        assert_eq!(
            v,
            SlideVerdict::Ok {
                offset: 200,
                phase: SlidePhase::Hidden,
            }
        );
    }

    #[test]
    fn open_at_end_offset_zero() {
        let v = compute(300, 300, 200, true);
        assert_eq!(
            v,
            SlideVerdict::Ok {
                offset: 0,
                phase: SlidePhase::FullyShown,
            }
        );
    }

    #[test]
    fn close_at_start_offset_zero() {
        let v = compute(0, 300, 200, false);
        assert_eq!(
            v,
            SlideVerdict::Ok {
                offset: 0,
                phase: SlidePhase::FullyShown,
            }
        );
    }

    #[test]
    fn close_at_end_offset_full() {
        let v = compute(300, 300, 200, false);
        assert_eq!(
            v,
            SlideVerdict::Ok {
                offset: 200,
                phase: SlidePhase::Hidden,
            }
        );
    }

    #[test]
    fn animating_in_middle() {
        let v = compute(150, 300, 200, true);
        if let SlideVerdict::Ok { phase, .. } = v {
            assert_eq!(phase, SlidePhase::Animating);
        }
    }

    #[test]
    fn opening_offset_decreases_over_time() {
        let v1 = compute(50, 300, 200, true);
        let v2 = compute(150, 300, 200, true);
        if let (SlideVerdict::Ok { offset: o1, .. }, SlideVerdict::Ok { offset: o2, .. }) = (v1, v2)
        {
            assert!(o2 < o1);
        }
    }

    #[test]
    fn closing_offset_increases_over_time() {
        let v1 = compute(50, 300, 200, false);
        let v2 = compute(150, 300, 200, false);
        if let (SlideVerdict::Ok { offset: o1, .. }, SlideVerdict::Ok { offset: o2, .. }) = (v1, v2)
        {
            assert!(o2 > o1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(150, 300, 200, true);
        let r2 = compute(150, 300, 200, true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offset_le_drawer_width() {
        let v = compute(150, 300, 200, true);
        if let SlideVerdict::Ok { offset, .. } = v {
            assert!(offset <= 200);
        }
    }

    #[test]
    fn elapsed_past_duration_clamps() {
        let v = compute(1000, 300, 200, true);
        assert_eq!(
            v,
            SlideVerdict::Ok {
                offset: 0,
                phase: SlidePhase::FullyShown,
            }
        );
    }

    #[test]
    fn ease_out_starts_fast() {
        // 25% time should be more than 25% animated (ease-out front-loaded).
        let v = compute(75, 300, 200, true);
        if let SlideVerdict::Ok { offset, .. } = v {
            // Linear at 25% would be 150 (0.75 * 200). Ease-out 25% → 0.578^3 → 1-0.422 = 0.578 progress closing → offset ≈ 200 * 0.422 = ~84.
            assert!(offset < 150);
        }
    }
}
