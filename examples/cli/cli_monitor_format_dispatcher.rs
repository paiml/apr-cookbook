//! # apr monitor — `--format` Dispatcher (tui / json / text)
//!
//! `apr monitor --format <FORMAT>` switches between three output modes:
//! `tui` (interactive), `json` (NDJSON for LLM agents and CI), and
//! `text` (plain stdout for log-style consumption). This recipe builds
//! the dispatcher and asserts the contract: only the three known formats
//! are accepted, `--json` flag is shorthand for `--format json` (must
//! reconcile when both supplied).
//!
//! Demonstrates the **MONITOR.5** recipe for PMAT-101 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MONITOR-002
//!
//! Run with: cargo run --example cli_monitor_format_dispatcher
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Tui,
    Json,
    Text,
}

impl OutputFormat {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "tui" => Some(OutputFormat::Tui),
            "json" => Some(OutputFormat::Json),
            "text" => Some(OutputFormat::Text),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum FormatVerdict {
    Resolved(OutputFormat),
    UnknownFormat(String),
    Conflict {
        flag_says_json: bool,
        format_says: OutputFormat,
    },
}

pub fn resolve_format(format_arg: Option<&str>, json_flag: bool) -> FormatVerdict {
    let from_format = match format_arg {
        Some(s) => match OutputFormat::from_str_strict(s) {
            Some(f) => Some(f),
            None => return FormatVerdict::UnknownFormat(s.into()),
        },
        None => None,
    };
    match (from_format, json_flag) {
        (None, false) => FormatVerdict::Resolved(OutputFormat::Tui), // CLI default
        (None, true) => FormatVerdict::Resolved(OutputFormat::Json),
        (Some(f), false) => FormatVerdict::Resolved(f),
        (Some(OutputFormat::Json), true) => FormatVerdict::Resolved(OutputFormat::Json),
        (Some(f), true) => FormatVerdict::Conflict {
            flag_says_json: true,
            format_says: f,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_format_dispatcher")?;

    let cases = [
        ("default", None, false),
        ("--format tui", Some("tui"), false),
        ("--json", None, true),
        ("--format json --json", Some("json"), true),
        ("--format text --json (conflict)", Some("text"), true),
        ("unknown format", Some("yaml"), false),
    ];

    for (label, fmt, json_flag) in cases {
        println!("{label:>32}  →  {:?}", resolve_format(fmt, json_flag));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_resolves_to_tui() {
        // No format flag, no --json → TUI (the CLI default).
        assert_eq!(
            resolve_format(None, false),
            FormatVerdict::Resolved(OutputFormat::Tui)
        );
    }

    #[test]
    fn json_flag_alone_resolves_to_json() {
        assert_eq!(
            resolve_format(None, true),
            FormatVerdict::Resolved(OutputFormat::Json)
        );
    }

    #[test]
    fn explicit_format_text_resolves() {
        assert_eq!(
            resolve_format(Some("text"), false),
            FormatVerdict::Resolved(OutputFormat::Text)
        );
    }

    #[test]
    fn json_flag_with_json_format_is_consistent() {
        // Specifying both --format json and --json is redundant but not a conflict.
        assert_eq!(
            resolve_format(Some("json"), true),
            FormatVerdict::Resolved(OutputFormat::Json)
        );
    }

    #[test]
    fn json_flag_with_text_format_is_conflict() {
        // Operator probably forgot to remove one of the flags; surface the conflict.
        let v = resolve_format(Some("text"), true);
        assert!(matches!(v, FormatVerdict::Conflict { .. }));
    }

    #[test]
    fn unknown_format_rejected() {
        assert!(matches!(
            resolve_format(Some("yaml"), false),
            FormatVerdict::UnknownFormat(_)
        ));
    }

    #[test]
    fn empty_format_rejected_as_unknown() {
        assert!(matches!(
            resolve_format(Some(""), false),
            FormatVerdict::UnknownFormat(_)
        ));
    }
}
