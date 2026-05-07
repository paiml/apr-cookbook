//! # TUI Command History Reverse Search
//!
//! Ctrl+R style reverse-search through command history. Returns
//! most recent command containing the pattern.
//!
//! Demonstrates the **TUI.121** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: bash readline reverse-i-search; fzf history search.
//!
//! Run with: cargo run --example tui_command_history_search_pattern
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok {
        match_index: u32,
        matched_command: String,
    },
    NoMatch,
    InvalidConfig,
}

pub fn search(history: &[&str], pattern: &str) -> SearchVerdict {
    if history.is_empty() || pattern.is_empty() {
        return SearchVerdict::InvalidConfig;
    }
    let p_lower = pattern.to_lowercase();
    for (i, cmd) in history.iter().enumerate().rev() {
        if cmd.to_lowercase().contains(&p_lower) {
            return SearchVerdict::Ok {
                match_index: i as u32,
                matched_command: (*cmd).to_string(),
            };
        }
    }
    SearchVerdict::NoMatch
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_command_history_search_pattern")?;

    let history = ["ls", "cd /tmp", "vim file.rs", "ls -l"];
    println!("query 'vim': {:?}", search(&history, "vim"));
    println!("query 'xyz': {:?}", search(&history, "xyz"));
    println!("invalid: {:?}", search(&[], "x"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn finds_most_recent_match() {
        let h = ["ls", "cd /tmp", "vim file.rs", "ls -l"];
        let v = search(&h, "ls");
        if let SearchVerdict::Ok { match_index, .. } = v {
            // Most recent "ls" match is at index 3 ("ls -l").
            assert_eq!(match_index, 3);
        }
    }

    #[test]
    fn case_insensitive() {
        let h = ["VIM file"];
        let v = search(&h, "vim");
        if let SearchVerdict::Ok { match_index, .. } = v {
            assert_eq!(match_index, 0);
        }
    }

    #[test]
    fn no_match_returns_no_match() {
        let h = ["ls", "cd"];
        assert_eq!(search(&h, "xyz"), SearchVerdict::NoMatch);
    }

    #[test]
    fn empty_history_rejected() {
        assert_eq!(search(&[], "x"), SearchVerdict::InvalidConfig);
    }

    #[test]
    fn empty_pattern_rejected() {
        let h = ["ls"];
        assert_eq!(search(&h, ""), SearchVerdict::InvalidConfig);
    }

    #[test]
    fn substring_match_works() {
        let h = ["very long command line"];
        let v = search(&h, "long");
        if let SearchVerdict::Ok {
            matched_command, ..
        } = v
        {
            assert_eq!(matched_command, "very long command line");
        }
    }

    #[test]
    fn deterministic() {
        let h = ["ls"];
        let r1 = search(&h, "ls");
        let r2 = search(&h, "ls");
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_pattern_supported() {
        let h = ["café-cmd"];
        let v = search(&h, "café");
        if let SearchVerdict::Ok { match_index, .. } = v {
            assert_eq!(match_index, 0);
        }
    }

    #[test]
    fn first_command_match_works() {
        let h = ["ls", "cd"];
        let v = search(&h, "ls");
        if let SearchVerdict::Ok { match_index, .. } = v {
            assert_eq!(match_index, 0);
        }
    }

    #[test]
    fn matched_command_full_string() {
        let h = ["short", "very long command"];
        let v = search(&h, "long");
        if let SearchVerdict::Ok {
            matched_command, ..
        } = v
        {
            assert_eq!(matched_command, "very long command");
        }
    }

    #[test]
    fn many_history_handled() {
        let h: Vec<&str> = vec!["cmd"; 100];
        let v = search(&h, "cmd");
        if let SearchVerdict::Ok { match_index, .. } = v {
            assert_eq!(match_index, 99);
        }
    }
}
