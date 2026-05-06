//! # apr tui — Pager Buffer Size Planner
//!
//! When showing long output (logs, tables) in a TUI pager, buffer N
//! lines for scroll-up. N too small: jump-back lost; N too big: memory
//! cost on huge logs. Auto-pick: max(viewport × 4, 256), capped at
//! 65,536 lines (~ 8 MiB at 128B/line). Page-up/down jumps = viewport
//! − 1 (overlap of 1 line for context). This recipe builds the planner.
//!
//! Demonstrates the **TUI.6** recipe for PMAT-122 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-001 + less(1) / more(1) buffer conventions
//!
//! Run with: cargo run --example cli_tui_pager_buffer_planner
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FLOOR: u32 = 256;
const CEILING: u32 = 65_536;

#[derive(Debug, PartialEq)]
pub enum BufferVerdict {
    Ok { buffer_lines: u32 },
    InvalidViewport,
}

pub fn plan_buffer(viewport_height: u32) -> BufferVerdict {
    if viewport_height == 0 {
        return BufferVerdict::InvalidViewport;
    }
    let raw = viewport_height.saturating_mul(4);
    let buffered = raw.clamp(FLOOR, CEILING);
    BufferVerdict::Ok {
        buffer_lines: buffered,
    }
}

pub fn page_jump(viewport_height: u32) -> u32 {
    if viewport_height <= 1 {
        return 1;
    }
    viewport_height - 1
}

#[derive(Debug, PartialEq)]
pub enum ScrollVerdict {
    Top,
    Middle { progress: f64 },
    Bottom,
}

pub fn classify_position(top_line: u64, viewport_height: u32, total_lines: u64) -> ScrollVerdict {
    if total_lines == 0 || viewport_height == 0 {
        return ScrollVerdict::Top;
    }
    let bottom = top_line + u64::from(viewport_height);
    if top_line == 0 {
        ScrollVerdict::Top
    } else if bottom >= total_lines {
        ScrollVerdict::Bottom
    } else {
        let progress =
            top_line as f64 / total_lines.saturating_sub(u64::from(viewport_height)) as f64;
        ScrollVerdict::Middle {
            progress: progress.clamp(0.0, 1.0),
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_pager_buffer_planner")?;

    for h in [0u32, 24, 50, 200, 32_000] {
        println!(
            "viewport={h}  buf={:?}  jump={}",
            plan_buffer(h),
            page_jump(h)
        );
    }
    for top in [0u64, 50, 950, 1000] {
        println!("top={top}  →  {:?}", classify_position(top, 24, 1000));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_viewport_uses_floor() {
        // 24 × 4 = 96 < 256 floor.
        let v = plan_buffer(24);
        assert_eq!(
            v,
            BufferVerdict::Ok {
                buffer_lines: FLOOR
            }
        );
    }

    #[test]
    fn medium_viewport_uses_4x() {
        // 100 × 4 = 400 > 256 floor.
        let v = plan_buffer(100);
        assert_eq!(v, BufferVerdict::Ok { buffer_lines: 400 });
    }

    #[test]
    fn huge_viewport_clamped_to_ceiling() {
        let v = plan_buffer(50_000);
        assert_eq!(
            v,
            BufferVerdict::Ok {
                buffer_lines: CEILING
            }
        );
    }

    #[test]
    fn zero_viewport_invalid() {
        assert_eq!(plan_buffer(0), BufferVerdict::InvalidViewport);
    }

    #[test]
    fn page_jump_one_line_overlap() {
        assert_eq!(page_jump(24), 23);
        assert_eq!(page_jump(50), 49);
    }

    #[test]
    fn page_jump_clamps_for_tiny_viewport() {
        assert_eq!(page_jump(1), 1);
        assert_eq!(page_jump(0), 1);
    }

    #[test]
    fn position_at_top() {
        assert_eq!(classify_position(0, 24, 1000), ScrollVerdict::Top);
    }

    #[test]
    fn position_at_bottom() {
        // top=976, viewport=24 → bottom = 1000 = total → Bottom.
        assert_eq!(classify_position(976, 24, 1000), ScrollVerdict::Bottom);
    }

    #[test]
    fn position_in_middle_returns_progress() {
        let v = classify_position(488, 24, 1000);
        if let ScrollVerdict::Middle { progress } = v {
            assert!((0.0..=1.0).contains(&progress));
            assert!(progress > 0.4 && progress < 0.6);
        }
    }

    #[test]
    fn empty_total_is_top() {
        assert_eq!(classify_position(0, 24, 0), ScrollVerdict::Top);
    }
}
