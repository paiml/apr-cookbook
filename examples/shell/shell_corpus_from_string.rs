//! # Shell — Build a Corpus from In-Memory Commands
//!
//! Turn a vector of shell commands (e.g., from `HistoryParser` or a synthetic
//! seed) into an `aprender_shell::corpus::Corpus`. The corpus is the input
//! to model training (Markov chains, paged-model n-grams).
//!
//! Demonstrates the **SH.2** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! second stage of the history → corpus → model pipeline.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Manning, C. D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. ISBN: 978-0521865715
//!
//! Run with: cargo run --example shell_corpus_from_string
//!
//! Added by PMAT-081 (expand-cookbooks: aprender-shell coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_shell::corpus::Corpus;

const SAMPLE_COMMANDS: &str = "\
ls -la
git status
cargo build --release
git commit -m wip
cargo test --all-features
git push origin main
ls -la
git log --oneline -5
";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_corpus_from_string")?;

    let corpus = Corpus::from_string(SAMPLE_COMMANDS)
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("corpus: {e}")))?;

    let stats = corpus.coverage_stats();
    println!("corpus has {} commands", corpus.len());
    println!("unique prefixes: {}", corpus.prefixes().len());
    println!("coverage stats: {stats:?}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn corpus_size_matches_input() {
        let corpus = Corpus::from_string(SAMPLE_COMMANDS).unwrap();
        // 8 non-empty lines in SAMPLE_COMMANDS.
        assert_eq!(corpus.len(), 8);
    }

    #[test]
    fn empty_input_is_rejected() {
        // Corpus::from_string("") returns Err(CorpusError::Empty) — the API
        // treats empty input as invalid rather than producing an empty corpus.
        let err = Corpus::from_string("");
        assert!(err.is_err(), "empty input should be rejected");
    }
}
