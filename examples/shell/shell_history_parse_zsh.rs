//! # Shell — Parse ZSH History (in-memory)
//!
//! Use `aprender_shell::history::HistoryParser` to parse a synthetic ZSH
//! extended-format history file from an in-memory string. The parser
//! handles the `: timestamp:0;command` ZSH wire format, plain bash format,
//! comment stripping, and shell-no-op filtering.
//!
//! Demonstrates the **SH.1** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the entry point of the history → corpus → model pipeline.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bourne, S. R. (1978). The UNIX Shell. Bell System Technical Journal 57(6). DOI: 10.1002/j.1538-7305.1978.tb02137.x
//!
//! Run with: cargo run --example shell_history_parse_zsh
//!
//! Added by PMAT-081 (expand-cookbooks: aprender-shell coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_shell::history::HistoryParser;
use std::io::Write;

const SAMPLE_ZSH_HISTORY: &str = "\
: 1700000000:0;ls -la
: 1700000060:0;git status
: 1700000120:0;cargo build --release
: 1700000180:0;git commit -m \"wip\"
: 1700000240:0;cargo test --all-features
: 1700000300:0;git push origin main
# this is a comment line, should be skipped
: 1700000360:0;ls -la
: 1700000420:0;git log --oneline -5
";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_history_parse_zsh")?;

    // Write the synthetic history to a tempfile and parse it.
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("zsh_history");
    let mut file = std::fs::File::create(&path)?;
    file.write_all(SAMPLE_ZSH_HISTORY.as_bytes())?;
    drop(file);

    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&path)
        .map_err(apr_cookbook::CookbookError::Io)?;

    println!(
        "parsed {} commands from synthetic ZSH history:",
        commands.len()
    );
    for (i, cmd) in commands.iter().take(5).enumerate() {
        println!("  {i}: {cmd}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn comment_lines_are_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zsh_history");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(SAMPLE_ZSH_HISTORY.as_bytes()).unwrap();
        drop(file);
        let parser = HistoryParser::new();
        let commands = parser.parse_file(&path).unwrap();
        // Comment line should NOT appear in parsed output.
        assert!(
            !commands.iter().any(|c| c.starts_with('#')),
            "comment lines must be filtered: {commands:?}"
        );
    }

    #[test]
    fn parses_known_command_count() {
        // 8 non-comment lines in SAMPLE_ZSH_HISTORY.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zsh_history");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(SAMPLE_ZSH_HISTORY.as_bytes()).unwrap();
        drop(file);
        let parser = HistoryParser::new();
        let commands = parser.parse_file(&path).unwrap();
        assert_eq!(commands.len(), 8);
    }
}
