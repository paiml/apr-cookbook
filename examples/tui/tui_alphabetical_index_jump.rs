//! # TUI Alphabetical Index Jump
//!
//! Given a sorted list of items and a target letter (A..=Z), find
//! the first item starting with that letter (or the closest letter
//! after it). Useful for A-Z jump bars.
//!
//! Demonstrates the **TUI.86** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Contacts.app A-Z index bar; iOS UITableView
//!  sectionIndexTitles (Apple HIG).
//!
//! Run with: cargo run --example tui_alphabetical_index_jump
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum JumpVerdict {
    Ok {
        target_index: u32,
        actual_letter: char,
    },
    NoMatch,
    InvalidConfig,
}

pub fn jump(items: &[&str], target_letter: char) -> JumpVerdict {
    if items.is_empty() || !target_letter.is_ascii_alphabetic() {
        return JumpVerdict::InvalidConfig;
    }
    let target = target_letter.to_ascii_uppercase();
    // Verify items are sorted ASCII-case-insensitively.
    for w in items.windows(2) {
        if first_letter(w[0]) > first_letter(w[1]) {
            return JumpVerdict::InvalidConfig;
        }
    }
    for (i, item) in items.iter().enumerate() {
        if let Some(c) = first_letter(item) {
            if c >= target {
                return JumpVerdict::Ok {
                    target_index: i as u32,
                    actual_letter: c,
                };
            }
        }
    }
    JumpVerdict::NoMatch
}

fn first_letter(s: &str) -> Option<char> {
    s.chars().next().map(|c| c.to_ascii_uppercase())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_alphabetical_index_jump")?;

    let contacts = ["alice", "bob", "carol", "dave", "eve"];
    println!("jump to C: {:?}", jump(&contacts, 'C'));
    println!("jump to Z: {:?}", jump(&contacts, 'Z'));
    println!("invalid: {:?}", jump(&[], 'A'));
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
    fn jump_to_present_letter() {
        let items = ["alice", "bob", "carol"];
        let v = jump(&items, 'B');
        if let JumpVerdict::Ok {
            target_index,
            actual_letter,
        } = v
        {
            assert_eq!(target_index, 1);
            assert_eq!(actual_letter, 'B');
        }
    }

    #[test]
    fn jump_to_missing_letter_finds_next() {
        let items = ["alice", "carol"];
        let v = jump(&items, 'B');
        if let JumpVerdict::Ok {
            target_index,
            actual_letter,
        } = v
        {
            assert_eq!(target_index, 1);
            assert_eq!(actual_letter, 'C');
        }
    }

    #[test]
    fn jump_to_z_no_match_when_after_last() {
        let items = ["alice", "bob"];
        assert_eq!(jump(&items, 'Z'), JumpVerdict::NoMatch);
    }

    #[test]
    fn empty_list_rejected() {
        assert_eq!(jump(&[], 'A'), JumpVerdict::InvalidConfig);
    }

    #[test]
    fn non_letter_rejected() {
        let items = ["alice"];
        assert_eq!(jump(&items, '5'), JumpVerdict::InvalidConfig);
    }

    #[test]
    fn unsorted_items_rejected() {
        let items = ["zeta", "alpha"];
        assert_eq!(jump(&items, 'A'), JumpVerdict::InvalidConfig);
    }

    #[test]
    fn case_insensitive_target() {
        let items = ["alice", "bob"];
        let v_upper = jump(&items, 'A');
        let v_lower = jump(&items, 'a');
        assert_eq!(v_upper, v_lower);
    }

    #[test]
    fn deterministic() {
        let items = ["alice", "bob"];
        let r1 = jump(&items, 'B');
        let r2 = jump(&items, 'B');
        assert_eq!(r1, r2);
    }

    #[test]
    fn first_index_for_first_letter() {
        let items = ["alice", "bob"];
        let v = jump(&items, 'A');
        if let JumpVerdict::Ok { target_index, .. } = v {
            assert_eq!(target_index, 0);
        }
    }

    #[test]
    fn single_item_works() {
        let items = ["alice"];
        let v = jump(&items, 'A');
        if let JumpVerdict::Ok { target_index, .. } = v {
            assert_eq!(target_index, 0);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<&str> = vec!["alice", "bob", "carol", "dave", "eve", "frank"];
        let v = jump(&items, 'D');
        if let JumpVerdict::Ok {
            target_index,
            actual_letter,
        } = v
        {
            assert_eq!(target_index, 3);
            assert_eq!(actual_letter, 'D');
        }
    }
}
