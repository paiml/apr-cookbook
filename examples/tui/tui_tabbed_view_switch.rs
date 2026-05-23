//! # TUI Tabbed View Switch
//!
//! Manage tabbed-view state: switch active tab via keyboard
//! (Ctrl+Tab, Ctrl+Shift+Tab, Ctrl+1..9). Returns next-state with
//! whether to redraw.
//!
//! Demonstrates the **TUI.153** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code tab-switch keybindings; Firefox `:tabnext` cycle
//!  semantics.
//!
//! Run with: cargo run --example tui_tabbed_view_switch
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TabVerdict {
    Ok { active_idx: u32, redraw: bool },
    InvalidConfig,
}

pub fn step(tab_count: u32, current_idx: u32, key: &str) -> TabVerdict {
    if tab_count == 0 || current_idx >= tab_count {
        return TabVerdict::InvalidConfig;
    }
    let next = match key {
        "Ctrl+Tab" => (current_idx + 1) % tab_count,
        "Ctrl+Shift+Tab" => (current_idx + tab_count - 1) % tab_count,
        k if k.starts_with("Ctrl+") => {
            let n_str = &k[5..];
            match n_str.parse::<u32>() {
                Ok(n) if (1..=9).contains(&n) && n <= tab_count => n - 1,
                _ => current_idx,
            }
        }
        _ => current_idx,
    };
    TabVerdict::Ok {
        active_idx: next,
        redraw: next != current_idx,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_tabbed_view_switch")?;

    println!("Ctrl+Tab: {:?}", step(3, 0, "Ctrl+Tab"));
    println!("Ctrl+3: {:?}", step(5, 0, "Ctrl+3"));
    println!("invalid: {:?}", step(0, 0, "Ctrl+Tab"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stepper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_tabs() {
        assert_eq!(step(0, 0, "Ctrl+Tab"), TabVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_idx_oob() {
        assert_eq!(step(2, 5, "Ctrl+Tab"), TabVerdict::InvalidConfig);
    }

    #[test]
    fn ctrl_tab_advances() {
        let v = step(3, 0, "Ctrl+Tab");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 1,
                redraw: true,
            }
        );
    }

    #[test]
    fn ctrl_tab_wraps() {
        let v = step(3, 2, "Ctrl+Tab");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 0,
                redraw: true
            }
        );
    }

    #[test]
    fn ctrl_shift_tab_back() {
        let v = step(3, 1, "Ctrl+Shift+Tab");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 0,
                redraw: true
            }
        );
    }

    #[test]
    fn ctrl_shift_tab_wraps_to_last() {
        let v = step(3, 0, "Ctrl+Shift+Tab");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 2,
                redraw: true
            }
        );
    }

    #[test]
    fn ctrl_n_jumps_to_index() {
        let v = step(5, 0, "Ctrl+3");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 2,
                redraw: true
            }
        );
    }

    #[test]
    fn ctrl_n_oob_keeps_current() {
        let v = step(3, 0, "Ctrl+9");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 0,
                redraw: false
            }
        );
    }

    #[test]
    fn unknown_key_no_redraw() {
        let v = step(3, 0, "X");
        assert_eq!(
            v,
            TabVerdict::Ok {
                active_idx: 0,
                redraw: false
            }
        );
    }

    #[test]
    fn deterministic() {
        let r1 = step(3, 0, "Ctrl+Tab");
        let r2 = step(3, 0, "Ctrl+Tab");
        assert_eq!(r1, r2);
    }

    #[test]
    fn ctrl_1_first_tab() {
        let v = step(5, 3, "Ctrl+1");
        if let TabVerdict::Ok { active_idx, .. } = v {
            assert_eq!(active_idx, 0);
        }
    }

    #[test]
    fn redraw_false_when_target_same() {
        // Ctrl+1 with current_idx already at 0
        let v = step(5, 0, "Ctrl+1");
        if let TabVerdict::Ok { redraw, .. } = v {
            assert!(!redraw);
        }
    }

    #[test]
    fn many_tabs_handled() {
        let v = step(20, 5, "Ctrl+Tab");
        if let TabVerdict::Ok { active_idx, .. } = v {
            assert_eq!(active_idx, 6);
        }
    }
}
