//! # TUI Input History Navigate
//!
//! Translate Up/Down arrow into history-navigation actions. Returns
//! the history entry to display (or None at boundary).
//!
//! Demonstrates the **TUI.126** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: bash readline previous-history/next-history; vim
//!  command-line history (q:).
//!
//! Run with: cargo run --example tui_input_history_navigate
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NavDirection {
    Older,
    Newer,
}

#[derive(Debug, PartialEq)]
pub enum NavVerdict {
    Ok {
        entry: Option<String>,
        new_index: u32,
    },
    InvalidConfig,
}

pub fn navigate(history: &[&str], current_index: u32, direction: NavDirection) -> NavVerdict {
    if history.is_empty() {
        return NavVerdict::InvalidConfig;
    }
    let n = history.len() as u32;
    let new_index = match direction {
        NavDirection::Older => {
            if current_index >= n {
                n - 1
            } else if current_index == 0 {
                0
            } else {
                current_index - 1
            }
        }
        NavDirection::Newer => {
            if current_index + 1 >= n {
                n
            } else {
                current_index + 1
            }
        }
    };
    let entry = if new_index < n {
        Some(history[new_index as usize].to_string())
    } else {
        None
    };
    NavVerdict::Ok { entry, new_index }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_input_history_navigate")?;

    let history = ["ls", "cd /tmp", "vim file.rs"];
    println!(
        "from end older: {:?}",
        navigate(&history, 3, NavDirection::Older)
    );
    println!(
        "from 0 newer: {:?}",
        navigate(&history, 0, NavDirection::Newer)
    );
    println!("invalid: {:?}", navigate(&[], 0, NavDirection::Older));
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
    fn older_decreases_index() {
        let h = ["a", "b", "c"];
        let v = navigate(&h, 2, NavDirection::Older);
        if let NavVerdict::Ok { new_index, .. } = v {
            assert_eq!(new_index, 1);
        }
    }

    #[test]
    fn newer_increases_index() {
        let h = ["a", "b", "c"];
        let v = navigate(&h, 0, NavDirection::Newer);
        if let NavVerdict::Ok { new_index, .. } = v {
            assert_eq!(new_index, 1);
        }
    }

    #[test]
    fn older_at_zero_stays() {
        let h = ["a"];
        let v = navigate(&h, 0, NavDirection::Older);
        if let NavVerdict::Ok { new_index, .. } = v {
            assert_eq!(new_index, 0);
        }
    }

    #[test]
    fn newer_past_end_returns_none() {
        let h = ["a", "b"];
        let v = navigate(&h, 1, NavDirection::Newer);
        if let NavVerdict::Ok { entry, .. } = v {
            assert!(entry.is_none());
        }
    }

    #[test]
    fn empty_history_rejected() {
        assert_eq!(
            navigate(&[], 0, NavDirection::Older),
            NavVerdict::InvalidConfig
        );
    }

    #[test]
    fn entry_returns_correct_string() {
        let h = ["alpha", "beta"];
        let v = navigate(&h, 1, NavDirection::Older);
        if let NavVerdict::Ok { entry, .. } = v {
            assert_eq!(entry, Some("alpha".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let h = ["a"];
        let r1 = navigate(&h, 0, NavDirection::Older);
        let r2 = navigate(&h, 0, NavDirection::Older);
        assert_eq!(r1, r2);
    }

    #[test]
    fn out_of_range_clamps_to_last() {
        let h = ["a", "b"];
        let v = navigate(&h, 10, NavDirection::Older);
        if let NavVerdict::Ok { new_index, .. } = v {
            assert_eq!(new_index, 1);
        }
    }

    #[test]
    fn single_entry_works() {
        let h = ["only"];
        let v = navigate(&h, 0, NavDirection::Older);
        if let NavVerdict::Ok { entry, .. } = v {
            assert_eq!(entry, Some("only".to_string()));
        }
    }

    #[test]
    fn round_trip_older_newer() {
        let h = ["a", "b", "c"];
        let v1 = navigate(&h, 2, NavDirection::Older);
        if let NavVerdict::Ok { new_index, .. } = v1 {
            let v2 = navigate(&h, new_index, NavDirection::Newer);
            if let NavVerdict::Ok { new_index: i2, .. } = v2 {
                assert_eq!(i2, 2);
            }
        }
    }

    #[test]
    fn many_entries_handled() {
        let h: Vec<&str> = vec!["x"; 100];
        let v = navigate(&h, 50, NavDirection::Older);
        if let NavVerdict::Ok { new_index, .. } = v {
            assert_eq!(new_index, 49);
        }
    }
}
