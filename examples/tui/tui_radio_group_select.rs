//! # TUI Radio Group Selection
//!
//! Manage exclusive selection within a radio-button group.
//! Selecting an option deselects the previously-selected one.
//!
//! Demonstrates the **TUI.97** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML <input type=radio> name-grouping; macOS Cocoa
//!  NSMatrix radio mode.
//!
//! Run with: cargo run --example tui_radio_group_select
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RadioVerdict {
    Ok {
        selected_index: u32,
        previous_index: Option<u32>,
    },
    OutOfBounds,
    InvalidConfig,
}

pub fn select(options: &[&str], current: Option<u32>, target: u32) -> RadioVerdict {
    if options.is_empty() {
        return RadioVerdict::InvalidConfig;
    }
    if (target as usize) >= options.len() {
        return RadioVerdict::OutOfBounds;
    }
    RadioVerdict::Ok {
        selected_index: target,
        previous_index: current,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_radio_group_select")?;

    let options = ["small", "medium", "large"];
    println!("first select: {:?}", select(&options, None, 0));
    println!("change select: {:?}", select(&options, Some(0), 2));
    println!("oob: {:?}", select(&options, None, 10));
    println!("invalid: {:?}", select(&[], None, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_select_no_previous() {
        let v = select(&["a", "b"], None, 0);
        if let RadioVerdict::Ok {
            selected_index,
            previous_index,
        } = v
        {
            assert_eq!(selected_index, 0);
            assert!(previous_index.is_none());
        }
    }

    #[test]
    fn change_records_previous() {
        let v = select(&["a", "b"], Some(0), 1);
        if let RadioVerdict::Ok {
            selected_index,
            previous_index,
        } = v
        {
            assert_eq!(selected_index, 1);
            assert_eq!(previous_index, Some(0));
        }
    }

    #[test]
    fn out_of_range_returns_oob() {
        assert_eq!(select(&["a"], None, 5), RadioVerdict::OutOfBounds);
    }

    #[test]
    fn empty_options_rejected() {
        assert_eq!(select(&[], None, 0), RadioVerdict::InvalidConfig);
    }

    #[test]
    fn select_same_keeps_index() {
        let v = select(&["a", "b"], Some(0), 0);
        if let RadioVerdict::Ok {
            selected_index,
            previous_index,
        } = v
        {
            assert_eq!(selected_index, 0);
            assert_eq!(previous_index, Some(0));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = select(&["a", "b"], Some(0), 1);
        let r2 = select(&["a", "b"], Some(0), 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn last_index_works() {
        let v = select(&["a", "b", "c"], None, 2);
        if let RadioVerdict::Ok { selected_index, .. } = v {
            assert_eq!(selected_index, 2);
        }
    }

    #[test]
    fn one_option_only_works() {
        let v = select(&["only"], None, 0);
        if let RadioVerdict::Ok { selected_index, .. } = v {
            assert_eq!(selected_index, 0);
        }
    }

    #[test]
    fn previous_preserved_in_chain() {
        let v1 = select(&["a", "b", "c"], None, 1);
        if let RadioVerdict::Ok { selected_index, .. } = v1 {
            let v2 = select(&["a", "b", "c"], Some(selected_index), 2);
            if let RadioVerdict::Ok {
                selected_index: s2,
                previous_index: p2,
            } = v2
            {
                assert_eq!(s2, 2);
                assert_eq!(p2, Some(1));
            }
        }
    }

    #[test]
    fn boundary_one_above_oob() {
        let v = select(&["a", "b"], None, 2);
        assert_eq!(v, RadioVerdict::OutOfBounds);
    }

    #[test]
    fn many_options_handled() {
        let opts: Vec<&str> = vec!["x"; 50];
        let v = select(&opts, None, 49);
        if let RadioVerdict::Ok { selected_index, .. } = v {
            assert_eq!(selected_index, 49);
        }
    }
}
