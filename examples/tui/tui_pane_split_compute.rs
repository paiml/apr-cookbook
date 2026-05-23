//! # TUI Pane Split Compute
//!
//! Compute pane sizes when splitting a tmux/vim-style window. Returns
//! resulting pane widths and the active pane's new index after the
//! split.
//!
//! Demonstrates the **TUI.148** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: tmux split-window `-p` percentage; vim `:vsp` even-split
//!  semantics.
//!
//! Run with: cargo run --example tui_pane_split_compute
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        pane_widths: Vec<u32>,
        active_pane_idx: u32,
    },
    InvalidConfig,
}

pub fn split(current_widths: &[u32], active_idx: u32, new_pct: u32) -> SplitVerdict {
    if current_widths.is_empty() || (active_idx as usize) >= current_widths.len() {
        return SplitVerdict::InvalidConfig;
    }
    if !(10..=90).contains(&new_pct) {
        return SplitVerdict::InvalidConfig;
    }
    let active_width = current_widths[active_idx as usize];
    if active_width < 4 {
        return SplitVerdict::InvalidConfig;
    }
    let new_width = (active_width * new_pct) / 100;
    let remainder = active_width - new_width;
    if new_width < 2 || remainder < 2 {
        return SplitVerdict::InvalidConfig;
    }
    let mut new_widths: Vec<u32> = Vec::with_capacity(current_widths.len() + 1);
    for (i, w) in current_widths.iter().enumerate() {
        if i as u32 == active_idx {
            new_widths.push(remainder);
            new_widths.push(new_width);
        } else {
            new_widths.push(*w);
        }
    }
    SplitVerdict::Ok {
        pane_widths: new_widths,
        active_pane_idx: active_idx + 1,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_pane_split_compute")?;

    println!("split 50/50: {:?}", split(&[80], 0, 50));
    println!("split 30/70: {:?}", split(&[80, 40], 1, 30));
    println!("invalid: {:?}", split(&[], 0, 50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(split(&[], 0, 50), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn idx_oob_rejected() {
        assert_eq!(split(&[80], 5, 50), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_pct_too_low() {
        assert_eq!(split(&[80], 0, 5), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_pct_too_high() {
        assert_eq!(split(&[80], 0, 95), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_narrow_pane() {
        assert_eq!(split(&[3], 0, 50), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn fifty_fifty_split() {
        let v = split(&[80], 0, 50);
        if let SplitVerdict::Ok { pane_widths, .. } = v {
            assert_eq!(pane_widths, vec![40, 40]);
        }
    }

    #[test]
    fn pane_count_increases_by_one() {
        let v = split(&[80, 40], 0, 50);
        if let SplitVerdict::Ok { pane_widths, .. } = v {
            assert_eq!(pane_widths.len(), 3);
        }
    }

    #[test]
    fn active_idx_advances() {
        let v = split(&[80], 0, 50);
        if let SplitVerdict::Ok {
            active_pane_idx, ..
        } = v
        {
            assert_eq!(active_pane_idx, 1);
        }
    }

    #[test]
    fn other_panes_unchanged() {
        let v = split(&[80, 40], 1, 50);
        if let SplitVerdict::Ok { pane_widths, .. } = v {
            assert_eq!(pane_widths[0], 80);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = split(&[80], 0, 50);
        let r2 = split(&[80], 0, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn total_width_preserved() {
        let v = split(&[80, 40], 0, 50);
        if let SplitVerdict::Ok { pane_widths, .. } = v {
            let total: u32 = pane_widths.iter().sum();
            assert_eq!(total, 120);
        }
    }

    #[test]
    fn many_panes_handled() {
        let widths: Vec<u32> = vec![20; 10];
        let v = split(&widths, 5, 50);
        if let SplitVerdict::Ok { pane_widths, .. } = v {
            assert_eq!(pane_widths.len(), 11);
        }
    }
}
