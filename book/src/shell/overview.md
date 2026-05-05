# Shell — `aprender-shell` AI-Powered Completion

Recipes for [`aprender-shell`](https://crates.io/crates/aprender-shell) v0.31.2 — lightweight shell-completion engine that trains a local `.apr` model from your shell history (zsh `.zsh_history`, bash `.bash_history`, fish `~/.local/share/fish/fish_history`) and proposes completions inline.

Closes the **≥3 recipes per sister crate** requirement from [expand-cookbooks/subcrate-coverage.md](../../specifications/expand-cookbooks/subcrate-coverage.md).

## Recipes

| # | Recipe | What |
|---|--------|------|
| SH.1 | [`shell_history_parse_zsh`](https://github.com/paiml/apr-cookbook/blob/main/examples/shell/shell_history_parse_zsh.rs) | Parse synthetic ZSH extended-format history via `HistoryParser`; comment-line filtering |
| SH.2 | [`shell_corpus_from_string`](https://github.com/paiml/apr-cookbook/blob/main/examples/shell/shell_corpus_from_string.rs) | `Corpus::from_string` with inline commands; coverage stats; empty input rejection |
| SH.3 | [`shell_trie_prefix_completion`](https://github.com/paiml/apr-cookbook/blob/main/examples/shell/shell_trie_prefix_completion.rs) | `Trie` prefix index with frequency ranking (top-K candidates) |

## API surface exercised

- `aprender_shell::history::HistoryParser` — ZSH extended + bash + fish formats
- `aprender_shell::corpus::Corpus::{from_string, coverage_stats}`
- `aprender_shell::trie::Trie::{insert, find_prefix}`

## Provenance

Added during PMAT-081 (expand-cookbooks initiative, v6.1.0).
