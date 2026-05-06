//! # TUI Command Palette Fuzzy Score
//!
//! Score commands by fuzzy substring match: each query character must
//! appear in order; tighter clusters score higher. Returns ranked
//! command list (highest score first).
//!
//! Demonstrates the **TUI.94** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: skim/fzf scoring algorithm; Sublime Goto Anything.
//!
//! Run with: cargo run --example tui_command_palette_score
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ScoreVerdict {
    Ok { ranked: Vec<(String, i32)> },
    InvalidConfig,
}

pub fn score(commands: &[&str], query: &str) -> ScoreVerdict {
    if commands.is_empty() || query.is_empty() {
        return ScoreVerdict::InvalidConfig;
    }
    let q_lower = query.to_lowercase();
    let mut ranked: Vec<(String, i32)> = Vec::new();
    for cmd in commands {
        let cmd_lower = cmd.to_lowercase();
        if let Some(s) = fuzzy_score(&cmd_lower, &q_lower) {
            ranked.push(((*cmd).to_string(), s));
        }
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ScoreVerdict::Ok { ranked }
}

fn fuzzy_score(haystack: &str, query: &str) -> Option<i32> {
    let h_chars: Vec<char> = haystack.chars().collect();
    let q_chars: Vec<char> = query.chars().collect();
    let mut score = 0i32;
    let mut h_idx = 0usize;
    let mut last_match: Option<usize> = None;
    for q_char in &q_chars {
        let mut found = false;
        while h_idx < h_chars.len() {
            if h_chars[h_idx].eq_ignore_ascii_case(q_char) {
                score += 10;
                if let Some(prev) = last_match {
                    let gap = (h_idx - prev) as i32;
                    score -= gap.saturating_sub(1);
                }
                last_match = Some(h_idx);
                h_idx += 1;
                found = true;
                break;
            }
            h_idx += 1;
        }
        if !found {
            return None;
        }
    }
    Some(score)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_command_palette_score")?;

    let commands = ["save_file", "save_all", "load_file", "exit"];
    println!("query 'sf': {:?}", score(&commands, "sf"));
    println!("query 'load': {:?}", score(&commands, "load"));
    println!("invalid: {:?}", score(&[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_top_score() {
        let v = score(&["save_file", "load_file"], "save_file");
        if let ScoreVerdict::Ok { ranked } = v {
            assert_eq!(ranked[0].0, "save_file");
        }
    }

    #[test]
    fn fuzzy_query_finds_command() {
        let v = score(&["save_file", "load_file", "exit"], "sf");
        if let ScoreVerdict::Ok { ranked } = v {
            assert!(ranked.iter().any(|(c, _)| c == "save_file"));
        }
    }

    #[test]
    fn no_match_no_result() {
        let v = score(&["save", "load"], "xyz");
        if let ScoreVerdict::Ok { ranked } = v {
            assert!(ranked.is_empty());
        }
    }

    #[test]
    fn case_insensitive() {
        let v = score(&["Save"], "SAVE");
        if let ScoreVerdict::Ok { ranked } = v {
            assert_eq!(ranked.len(), 1);
        }
    }

    #[test]
    fn empty_commands_rejected() {
        assert_eq!(score(&[], "a"), ScoreVerdict::InvalidConfig);
    }

    #[test]
    fn empty_query_rejected() {
        let cmds = ["a"];
        assert_eq!(score(&cmds, ""), ScoreVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = score(&["save_file"], "sf");
        let r2 = score(&["save_file"], "sf");
        assert_eq!(r1, r2);
    }

    #[test]
    fn closer_match_higher_score() {
        let v = score(&["sf_widget", "save_file"], "sf");
        if let ScoreVerdict::Ok { ranked } = v {
            // sf_widget has tighter cluster.
            assert_eq!(ranked[0].0, "sf_widget");
        }
    }

    #[test]
    fn ranked_descending() {
        let cmds = ["save_file", "save", "exit"];
        let v = score(&cmds, "sa");
        if let ScoreVerdict::Ok { ranked } = v {
            for w in ranked.windows(2) {
                assert!(w[0].1 >= w[1].1);
            }
        }
    }

    #[test]
    fn score_field_positive_on_match() {
        let v = score(&["save"], "s");
        if let ScoreVerdict::Ok { ranked } = v {
            assert!(ranked[0].1 > 0);
        }
    }

    #[test]
    fn single_char_match() {
        let v = score(&["save"], "a");
        if let ScoreVerdict::Ok { ranked } = v {
            assert_eq!(ranked.len(), 1);
        }
    }
}
