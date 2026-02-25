# APR TUI

Simulates a terminal UI for interactive model exploration, rendered in headless mode. Mirrors `apr tui` with 4 tabs: Overview, Tensors, Stats, and Help. Navigation between tabs is simulated without actual terminal rendering.

## CLI Equivalent
```bash
apr tui model.apr
```

## Key Concepts
- Tabbed model explorer (Overview, Tensors, Stats, Help)
- Headless TUI simulation for CI/testing
- Interactive model metadata browsing

## Run
```bash
cargo run --example cli_apr_tui
```

## Source
[`examples/cli/cli_apr_tui.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/cli/cli_apr_tui.rs)
