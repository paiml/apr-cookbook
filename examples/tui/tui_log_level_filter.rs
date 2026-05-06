//! # TUI Log Level Filter
//!
//! Filter log lines by minimum severity. Levels are
//! TRACE < DEBUG < INFO < WARN < ERROR < FATAL.
//! Returns kept lines with their parsed level + counts per level.
//!
//! Demonstrates the **TUI.67** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: log4j 2 LogLevel hierarchy; RFC 5424 syslog severities.
//!
//! Run with: cargo run --example tui_log_level_filter
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Ord, Eq)]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok {
        kept: Vec<(Level, String)>,
        counts: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

fn parse_level(line: &str) -> Option<Level> {
    let line = line.trim_start();
    if line.starts_with("[TRACE]") || line.starts_with("TRACE ") {
        Some(Level::Trace)
    } else if line.starts_with("[DEBUG]") || line.starts_with("DEBUG ") {
        Some(Level::Debug)
    } else if line.starts_with("[INFO]") || line.starts_with("INFO ") {
        Some(Level::Info)
    } else if line.starts_with("[WARN]") || line.starts_with("WARN ") {
        Some(Level::Warn)
    } else if line.starts_with("[ERROR]") || line.starts_with("ERROR ") {
        Some(Level::Error)
    } else if line.starts_with("[FATAL]") || line.starts_with("FATAL ") {
        Some(Level::Fatal)
    } else {
        None
    }
}

pub fn filter(lines: &[&str], min_level: Level) -> FilterVerdict {
    if lines.is_empty() {
        return FilterVerdict::InvalidConfig;
    }
    let mut kept: Vec<(Level, String)> = Vec::new();
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for line in lines {
        if let Some(lvl) = parse_level(line) {
            let key = format!("{lvl:?}");
            *counts.entry(key).or_insert(0) += 1;
            if lvl >= min_level {
                kept.push((lvl, (*line).to_string()));
            }
        }
    }
    FilterVerdict::Ok { kept, counts }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_log_level_filter")?;

    let lines = [
        "[INFO] starting",
        "[DEBUG] config loaded",
        "[ERROR] failed to connect",
        "[TRACE] poll loop tick",
    ];
    println!("min=Warn: {:?}", filter(&lines, Level::Warn));
    println!("min=Trace: {:?}", filter(&lines, Level::Trace));
    println!("invalid: {:?}", filter(&[], Level::Info));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn warn_threshold_drops_below() {
        let lines = ["[INFO] x", "[WARN] y", "[ERROR] z"];
        let v = filter(&lines, Level::Warn);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 2);
            assert!(kept.iter().all(|(l, _)| *l >= Level::Warn));
        }
    }

    #[test]
    fn trace_threshold_keeps_all() {
        let lines = ["[INFO] x", "[DEBUG] y", "[TRACE] z"];
        let v = filter(&lines, Level::Trace);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 3);
        }
    }

    #[test]
    fn unknown_lines_skipped() {
        let lines = ["random line", "[INFO] real"];
        let v = filter(&lines, Level::Trace);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 1);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(filter(&[], Level::Info), FilterVerdict::InvalidConfig);
    }

    #[test]
    fn counts_accurate() {
        let lines = ["[INFO] a", "[INFO] b", "[ERROR] c"];
        let v = filter(&lines, Level::Trace);
        if let FilterVerdict::Ok { counts, .. } = v {
            assert_eq!(counts.get("Info"), Some(&2));
            assert_eq!(counts.get("Error"), Some(&1));
        }
    }

    #[test]
    fn level_ordering() {
        assert!(Level::Fatal > Level::Error);
        assert!(Level::Error > Level::Warn);
        assert!(Level::Warn > Level::Info);
        assert!(Level::Info > Level::Debug);
        assert!(Level::Debug > Level::Trace);
    }

    #[test]
    fn fatal_threshold_only_fatal() {
        let lines = ["[ERROR] e", "[FATAL] f"];
        let v = filter(&lines, Level::Fatal);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 1);
            assert_eq!(kept[0].0, Level::Fatal);
        }
    }

    #[test]
    fn deterministic() {
        let lines = ["[INFO] a"];
        let r1 = filter(&lines, Level::Info);
        let r2 = filter(&lines, Level::Info);
        assert_eq!(r1, r2);
    }

    #[test]
    fn space_separator_form_supported() {
        let lines = ["INFO real"];
        let v = filter(&lines, Level::Info);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 1);
        }
    }

    #[test]
    fn leading_whitespace_tolerated() {
        let lines = ["   [INFO] padded"];
        let v = filter(&lines, Level::Info);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert_eq!(kept.len(), 1);
        }
    }

    #[test]
    fn line_content_preserved() {
        let lines = ["[INFO] my_message"];
        let v = filter(&lines, Level::Info);
        if let FilterVerdict::Ok { kept, .. } = v {
            assert!(kept[0].1.contains("my_message"));
        }
    }
}
