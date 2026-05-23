//! # TUI Command Palette Ranker
//!
//! Rank command-palette items by fuzzy-match score against a query.
//! Score: matches are weighted by position (early chars worth more)
//! and contiguity (adjacent matches get a bonus).
//!
//! Demonstrates the **TUI.16** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VSCode command palette + sublime-text fuzzy ranking.
//!
//! Run with: cargo run --example tui_command_palette
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PaletteVerdict {
    Ok { ranked: Vec<(u32, String)> },
    EmptyCommands,
    EmptyQuery,
}

pub fn rank(commands: &[&str], query: &str) -> PaletteVerdict {
    if commands.is_empty() {
        return PaletteVerdict::EmptyCommands;
    }
    if query.is_empty() {
        return PaletteVerdict::EmptyQuery;
    }
    let q: Vec<char> = query.chars().collect();
    let mut scored: Vec<(u32, &str)> = Vec::new();
    for cmd in commands {
        if let Some(mut score) = score_match(cmd, &q) {
            if cmd.eq_ignore_ascii_case(query) {
                score += 1000;
            }
            scored.push((score, *cmd));
        }
    }
    scored.sort_by_key(|b| std::cmp::Reverse(b.0));
    PaletteVerdict::Ok {
        ranked: scored
            .into_iter()
            .map(|(s, c)| (s, c.to_string()))
            .collect(),
    }
}

fn score_match(cmd: &str, q: &[char]) -> Option<u32> {
    let cmd_chars: Vec<char> = cmd.chars().collect();
    let mut qi = 0usize;
    let mut score: u32 = 0;
    let mut last_match: Option<usize> = None;
    for (i, c) in cmd_chars.iter().enumerate() {
        if qi >= q.len() {
            break;
        }
        if c.eq_ignore_ascii_case(&q[qi]) {
            // Earlier matches worth more.
            let pos_bonus = (cmd_chars.len() - i) as u32;
            score += pos_bonus;
            // Contiguity bonus.
            if let Some(prev) = last_match {
                if i == prev + 1 {
                    score += 5;
                }
            }
            last_match = Some(i);
            qi += 1;
        }
    }
    if qi == q.len() {
        Some(score)
    } else {
        None
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_command_palette")?;

    let cmds = [
        "open file",
        "open folder",
        "save",
        "search files",
        "find next",
    ];
    println!("query=of: {:?}", rank(&cmds, "of"));
    println!("query=fnd: {:?}", rank(&cmds, "fnd"));
    println!("empty query: {:?}", rank(&cmds, ""));
    println!("empty cmds: {:?}", rank(&[], "x"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_ranks_high() {
        let v = rank(&["save", "save_as", "open"], "save");
        if let PaletteVerdict::Ok { ranked } = v {
            assert_eq!(ranked[0].1, "save");
        }
    }

    #[test]
    fn fuzzy_subseq_match() {
        let v = rank(&["open file", "save file"], "of");
        if let PaletteVerdict::Ok { ranked } = v {
            assert!(ranked.iter().any(|(_, c)| c == "open file"));
        }
    }

    #[test]
    fn no_match_excluded() {
        let v = rank(&["save"], "xyz");
        if let PaletteVerdict::Ok { ranked } = v {
            assert!(ranked.is_empty());
        }
    }

    #[test]
    fn empty_commands_rejected() {
        assert_eq!(rank(&[], "x"), PaletteVerdict::EmptyCommands);
    }

    #[test]
    fn empty_query_rejected() {
        assert_eq!(rank(&["a"], ""), PaletteVerdict::EmptyQuery);
    }

    #[test]
    fn case_insensitive() {
        let v = rank(&["Save File"], "sf");
        if let PaletteVerdict::Ok { ranked } = v {
            assert!(!ranked.is_empty());
        }
    }

    #[test]
    fn contiguous_outranks_split() {
        let v = rank(&["abc", "a-b-c"], "ab");
        if let PaletteVerdict::Ok { ranked } = v {
            // "abc" should rank higher (contiguity bonus).
            assert_eq!(ranked[0].1, "abc");
        }
    }

    #[test]
    fn early_match_outranks_late() {
        let v = rank(&["x apple", "applex"], "ap");
        if let PaletteVerdict::Ok { ranked } = v {
            // "applex" matches earlier.
            assert_eq!(ranked[0].1, "applex");
        }
    }

    #[test]
    fn unicode_query() {
        let v = rank(&["café", "tea"], "café");
        if let PaletteVerdict::Ok { ranked } = v {
            assert_eq!(ranked[0].1, "café");
        }
    }

    #[test]
    fn deterministic() {
        let a = rank(&["save", "open"], "s");
        let b = rank(&["save", "open"], "s");
        assert_eq!(a, b);
    }
}
