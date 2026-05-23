//! # TUI Column Header Sort Cycle
//!
//! Cycle column-header sort state on click: None → Ascending →
//! Descending → None. Returns rendered string with arrow and the
//! next-state.
//!
//! Demonstrates the **TUI.159** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ARIA-1.2 `aria-sort` attribute states; HTML5
//!  `<th aria-sort="ascending">` rendering.
//!
//! Run with: cargo run --example tui_column_header_sort_cycle
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SortState {
    None,
    Ascending,
    Descending,
}

#[derive(Debug, PartialEq)]
pub enum SortCycleVerdict {
    Ok {
        rendered: String,
        next_state: SortState,
    },
    InvalidConfig,
}

pub fn cycle(header: &str, state: SortState) -> SortCycleVerdict {
    if header.is_empty() {
        return SortCycleVerdict::InvalidConfig;
    }
    let (indicator, next) = match state {
        SortState::None => (" ", SortState::Ascending),
        SortState::Ascending => ("↑", SortState::Descending),
        SortState::Descending => ("↓", SortState::None),
    };
    SortCycleVerdict::Ok {
        rendered: format!("{header} {indicator}"),
        next_state: next,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_column_header_sort_cycle")?;

    println!("none: {:?}", cycle("Name", SortState::None));
    println!("asc: {:?}", cycle("Name", SortState::Ascending));
    println!("desc: {:?}", cycle("Name", SortState::Descending));
    println!("invalid: {:?}", cycle("", SortState::None));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn none_to_asc() {
        let v = cycle("Col", SortState::None);
        if let SortCycleVerdict::Ok { next_state, .. } = v {
            assert_eq!(next_state, SortState::Ascending);
        }
    }

    #[test]
    fn asc_to_desc() {
        let v = cycle("Col", SortState::Ascending);
        if let SortCycleVerdict::Ok { next_state, .. } = v {
            assert_eq!(next_state, SortState::Descending);
        }
    }

    #[test]
    fn desc_to_none() {
        let v = cycle("Col", SortState::Descending);
        if let SortCycleVerdict::Ok { next_state, .. } = v {
            assert_eq!(next_state, SortState::None);
        }
    }

    #[test]
    fn empty_header_rejected() {
        assert_eq!(cycle("", SortState::None), SortCycleVerdict::InvalidConfig);
    }

    #[test]
    fn rendered_starts_with_header() {
        let v = cycle("Name", SortState::None);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with("Name"));
        }
    }

    #[test]
    fn asc_arrow_present() {
        let v = cycle("X", SortState::Ascending);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("↑"));
        }
    }

    #[test]
    fn desc_arrow_present() {
        let v = cycle("X", SortState::Descending);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("↓"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = cycle("X", SortState::None);
        let r2 = cycle("X", SortState::None);
        assert_eq!(r1, r2);
    }

    #[test]
    fn three_state_cycle_returns_to_start() {
        let v1 = cycle("X", SortState::None);
        if let SortCycleVerdict::Ok { next_state: s1, .. } = v1 {
            let v2 = cycle("X", s1);
            if let SortCycleVerdict::Ok { next_state: s2, .. } = v2 {
                let v3 = cycle("X", s2);
                if let SortCycleVerdict::Ok { next_state: s3, .. } = v3 {
                    assert_eq!(s3, SortState::None);
                }
            }
        }
    }

    #[test]
    fn unicode_header_supported() {
        let v = cycle("café", SortState::Ascending);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn long_header_handled() {
        let header = "Very Long Column Header With Spaces";
        let v = cycle(header, SortState::Ascending);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains(header));
        }
    }

    #[test]
    fn separator_space_between_header_arrow() {
        let v = cycle("X", SortState::Ascending);
        if let SortCycleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains(" "));
        }
    }
}
