//! # TUI Command Line History Dedupe
//!
//! Compact a shell-style history list: collapse consecutive
//! duplicates and keep only the most-recent occurrence overall.
//! Returns deduped list (most-recent-last preserved order).
//!
//! Demonstrates the **TUI.77** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: zsh `HIST_IGNORE_ALL_DUPS`; bash `HISTCONTROL=ignoredups`.
//!
//! Run with: cargo run --example tui_command_line_history_dedupe
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum HistoryVerdict {
    Ok {
        deduped: Vec<String>,
        removed_count: u32,
    },
    InvalidConfig,
}

pub fn dedupe(commands: &[&str]) -> HistoryVerdict {
    if commands.is_empty() {
        return HistoryVerdict::InvalidConfig;
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut deduped: Vec<String> = Vec::new();
    // Iterate in reverse, keep first occurrence (most recent), reverse back.
    for cmd in commands.iter().rev() {
        if seen.insert((*cmd).to_string()) {
            deduped.push((*cmd).to_string());
        }
    }
    deduped.reverse();
    let removed_count = (commands.len() - deduped.len()) as u32;
    HistoryVerdict::Ok {
        deduped,
        removed_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_command_line_history_dedupe")?;

    let cmds = ["ls", "cd /tmp", "ls", "vim foo.rs", "ls"];
    println!("dedupe: {:?}", dedupe(&cmds));
    println!("invalid: {:?}", dedupe(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_duplicates_unchanged() {
        let cmds = ["ls", "cd"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok {
            deduped,
            removed_count,
        } = v
        {
            assert_eq!(deduped, vec!["ls".to_string(), "cd".to_string()]);
            assert_eq!(removed_count, 0);
        }
    }

    #[test]
    fn keeps_most_recent_occurrence() {
        let cmds = ["ls", "cd", "ls"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            // After dedup, "ls" should appear only once, at position of last occurrence.
            assert_eq!(deduped, vec!["cd".to_string(), "ls".to_string()]);
        }
    }

    #[test]
    fn removed_count_accurate() {
        let cmds = ["ls", "ls", "ls", "ls"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { removed_count, .. } = v {
            assert_eq!(removed_count, 3);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(dedupe(&[]), HistoryVerdict::InvalidConfig);
    }

    #[test]
    fn single_command_kept() {
        let cmds = ["ls"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok {
            deduped,
            removed_count,
        } = v
        {
            assert_eq!(deduped, vec!["ls".to_string()]);
            assert_eq!(removed_count, 0);
        }
    }

    #[test]
    fn deterministic() {
        let cmds = ["a", "b", "a"];
        let r1 = dedupe(&cmds);
        let r2 = dedupe(&cmds);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let cmds = ["LS", "ls"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            assert_eq!(deduped.len(), 2);
        }
    }

    #[test]
    fn whitespace_distinguishes() {
        let cmds = ["ls ", "ls"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            assert_eq!(deduped.len(), 2);
        }
    }

    #[test]
    fn mid_sequence_dedup_keeps_later() {
        let cmds = ["x", "y", "z", "y"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            // Expected order: ["x", "z", "y"] — y is preserved at last occurrence.
            assert_eq!(
                deduped,
                vec!["x".to_string(), "z".to_string(), "y".to_string()]
            );
        }
    }

    #[test]
    fn unicode_command_supported() {
        let cmds = ["café", "café"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            assert_eq!(deduped, vec!["café".to_string()]);
        }
    }

    #[test]
    fn many_unique_commands_unchanged() {
        let cmds: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let v = dedupe(&cmds);
        if let HistoryVerdict::Ok { deduped, .. } = v {
            assert_eq!(deduped.len(), 5);
        }
    }
}
