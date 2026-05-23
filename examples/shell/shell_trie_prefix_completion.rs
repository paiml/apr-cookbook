//! # Shell — Trie-Based Prefix Completion
//!
//! Use `aprender_shell::trie::Trie` to build a prefix index of shell
//! commands and serve top-K completions ranked by frequency. This is the
//! lightweight alternative to a full Markov-chain model when prefix
//! frequency is the only signal you need.
//!
//! Demonstrates the **SH.3** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the simplest possible completion serve loop.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Fredkin, E. (1960). Trie memory. CACM 3(9). DOI: 10.1145/367390.367400
//!
//! Run with: cargo run --example shell_trie_prefix_completion
//!
//! Added by PMAT-081 (expand-cookbooks: aprender-shell coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_shell::trie::Trie;

const COMMANDS: &[&str] = &[
    "git status",
    "git status",
    "git status",
    "git checkout -b foo",
    "git checkout main",
    "git commit -m wip",
    "git commit -m fix",
    "git push origin main",
    "git pull --rebase",
    "git log --oneline",
    "cargo build --release",
    "cargo test --all-features",
    "cargo run --example basic_loading",
];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_trie_prefix_completion")?;

    let mut trie = Trie::new();
    for cmd in COMMANDS {
        trie.insert(cmd);
    }

    let prefixes = ["git ", "cargo ", "git che"];
    for prefix in &prefixes {
        let completions = trie.find_prefix(prefix, 3);
        println!("top-3 completions for {prefix:?}:");
        for (i, c) in completions.iter().enumerate() {
            println!("  {i}: {c}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn git_status_is_top_completion() {
        // "git status" appears 3 times in COMMANDS, more than any other "git "
        // prefix entry, so it should rank first.
        let mut trie = Trie::new();
        for cmd in COMMANDS {
            trie.insert(cmd);
        }
        let completions = trie.find_prefix("git ", 5);
        assert!(!completions.is_empty(), "should find git completions");
        assert_eq!(completions[0], "git status");
    }

    #[test]
    fn limit_is_respected() {
        let mut trie = Trie::new();
        for cmd in COMMANDS {
            trie.insert(cmd);
        }
        let completions = trie.find_prefix("git ", 2);
        assert!(completions.len() <= 2);
    }
}
