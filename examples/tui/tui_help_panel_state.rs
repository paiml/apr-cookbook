//! # TUI Help Panel State
//!
//! Toggle / page state for a help panel: open at page 0, navigate
//! pages with PgUp/PgDn, close with Esc. Returns the panel state.
//!
//! Demonstrates the **TUI.38** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: man-page pager / GNU info navigation.
//!
//! Run with: cargo run --example tui_help_panel_state
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpOp {
    Toggle,
    PageUp,
    PageDown,
    Escape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpState {
    pub open: bool,
    pub page: u32,
    pub max_pages: u32,
}

#[derive(Debug, PartialEq)]
pub enum HelpVerdict {
    Ok { state: HelpState },
    InvalidConfig,
}

pub fn step(state: &HelpState, op: HelpOp) -> HelpVerdict {
    if state.max_pages == 0 {
        return HelpVerdict::InvalidConfig;
    }
    let new = match op {
        HelpOp::Toggle => HelpState {
            open: !state.open,
            page: 0,
            ..state.clone()
        },
        HelpOp::PageUp => HelpState {
            page: state.page.saturating_sub(1),
            ..state.clone()
        },
        HelpOp::PageDown => HelpState {
            page: (state.page + 1).min(state.max_pages - 1),
            ..state.clone()
        },
        HelpOp::Escape => HelpState {
            open: false,
            ..state.clone()
        },
    };
    HelpVerdict::Ok { state: new }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_help_panel_state")?;

    let s0 = HelpState {
        open: false,
        page: 0,
        max_pages: 5,
    };
    let s1 = if let HelpVerdict::Ok { state } = step(&s0, HelpOp::Toggle) {
        state
    } else {
        s0.clone()
    };
    println!("toggled open: {s1:?}");
    let s2 = if let HelpVerdict::Ok { state } = step(&s1, HelpOp::PageDown) {
        state
    } else {
        s1.clone()
    };
    println!("page down: {s2:?}");
    let _s3 = if let HelpVerdict::Ok { state } = step(&s2, HelpOp::Escape) {
        state
    } else {
        s2.clone()
    };
    println!("escape closes");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed() -> HelpState {
        HelpState {
            open: false,
            page: 0,
            max_pages: 5,
        }
    }

    #[test]
    fn stepper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn toggle_opens() {
        let v = step(&closed(), HelpOp::Toggle);
        if let HelpVerdict::Ok { state } = v {
            assert!(state.open);
        }
    }

    #[test]
    fn toggle_closes_when_open() {
        let mut s = closed();
        s.open = true;
        let v = step(&s, HelpOp::Toggle);
        if let HelpVerdict::Ok { state } = v {
            assert!(!state.open);
        }
    }

    #[test]
    fn page_down_advances() {
        let mut s = closed();
        s.open = true;
        let v = step(&s, HelpOp::PageDown);
        if let HelpVerdict::Ok { state } = v {
            assert_eq!(state.page, 1);
        }
    }

    #[test]
    fn page_up_at_zero_clamps() {
        let v = step(&closed(), HelpOp::PageUp);
        if let HelpVerdict::Ok { state } = v {
            assert_eq!(state.page, 0);
        }
    }

    #[test]
    fn page_down_at_max_clamps() {
        let mut s = closed();
        s.page = 4;
        let v = step(&s, HelpOp::PageDown);
        if let HelpVerdict::Ok { state } = v {
            assert_eq!(state.page, 4);
        }
    }

    #[test]
    fn escape_closes() {
        let mut s = closed();
        s.open = true;
        let v = step(&s, HelpOp::Escape);
        if let HelpVerdict::Ok { state } = v {
            assert!(!state.open);
        }
    }

    #[test]
    fn invalid_zero_max_pages() {
        let mut s = closed();
        s.max_pages = 0;
        assert_eq!(step(&s, HelpOp::Toggle), HelpVerdict::InvalidConfig);
    }

    #[test]
    fn toggle_resets_page() {
        let mut s = closed();
        s.page = 3;
        let v = step(&s, HelpOp::Toggle);
        if let HelpVerdict::Ok { state } = v {
            assert_eq!(state.page, 0);
        }
    }

    #[test]
    fn escape_preserves_page() {
        let mut s = closed();
        s.open = true;
        s.page = 3;
        let v = step(&s, HelpOp::Escape);
        if let HelpVerdict::Ok { state } = v {
            assert_eq!(state.page, 3);
        }
    }

    #[test]
    fn deterministic() {
        let s = closed();
        let a = step(&s, HelpOp::Toggle);
        let b = step(&s, HelpOp::Toggle);
        assert_eq!(a, b);
    }
}
