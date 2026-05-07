//! # Recipe: TUI Log Tail with Filters
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr tui --view logs --tail --filter level=ERROR`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example tui_log_tail` exits 0
//! 2. [x] `cargo test --example tui_log_tail` passes
//! 3. [x] Deterministic output (fixed log lines)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr tui --tail` in-process (no async reader)
//! 10. [x] Unit tests cover level filter, substring filter, tail window
//!
//! ## Learning Objective
//! Demonstrates a log-tail widget: stream lines from a bounded buffer, apply
//! level + substring filters, and keep only the last N that match. Mirrors
//! the TUI mode `apr tui --tail` uses for live server logs.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tui_log_tail
//! ```
//!
//! ## References
//! - Dean, J. & Ghemawat, S. (2008). *MapReduce: Simplified Data Processing on Large Clusters*. CACM. DOI: 10.1145/1327452.1327492

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogLine {
    pub ts_ms: u64,
    pub level: Level,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct LogFilter {
    pub min_level: Option<Level>,
    pub substr: Option<String>,
}

pub fn filter_lines<'a>(lines: &'a [LogLine], f: &LogFilter) -> Vec<&'a LogLine> {
    lines
        .iter()
        .filter(|l| {
            f.min_level.map_or(true, |min| l.level >= min)
                && f.substr.as_deref().map_or(true, |s| l.message.contains(s))
        })
        .collect()
}

pub fn tail_n<'a>(lines: &[&'a LogLine], n: usize) -> Vec<&'a LogLine> {
    let start = lines.len().saturating_sub(n);
    lines[start..].to_vec()
}

pub fn demo_log() -> Vec<LogLine> {
    let mut lines = Vec::new();
    let messages: &[(Level, &str)] = &[
        (Level::Info, "server started on :8080"),
        (Level::Debug, "config loaded: batch=32"),
        (Level::Info, "worker ready pid=12345"),
        (Level::Warn, "slow response 520ms"),
        (Level::Error, "upstream timeout after 30s"),
        (Level::Info, "request ok id=abc"),
        (Level::Warn, "cache miss ratio 0.43"),
        (Level::Error, "disk write failed: ENOSPC"),
        (Level::Debug, "gc run took 12ms"),
        (Level::Info, "shutdown signal received"),
    ];
    for (i, (lvl, msg)) in messages.iter().enumerate() {
        lines.push(LogLine {
            ts_ms: (i as u64) * 150,
            level: *lvl,
            message: (*msg).to_string(),
        });
    }
    lines
}

pub fn render_tail(lines: &[&LogLine]) -> String {
    let mut s = String::from("+--------------------- LOG TAIL ---------------------+\n");
    for l in lines {
        s.push_str(&format!(
            "| [{:>5}] {:>7}ms {:<30} |\n",
            l.level.label(),
            l.ts_ms,
            truncate(&l.message, 30)
        ));
    }
    s.push_str("+----------------------------------------------------+\n");
    s
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("tui_log_tail")?;
    println!("=== Recipe: {} ===", ctx.name());

    let all = demo_log();
    let filter = LogFilter {
        min_level: Some(Level::Warn),
        substr: None,
    };
    let filtered = filter_lines(&all, &filter);
    let tail = tail_n(&filtered, 5);

    println!("Total lines:    {}", all.len());
    println!("After filter:   {}", filtered.len());
    println!("Tail (last 5):  {}", tail.len());
    println!("\n{}", render_tail(&tail));

    let report = json!({
        "recipe": ctx.name(),
        "n_total": all.len(),
        "n_filtered": filtered.len(),
        "n_tail": tail.len(),
        "filter": {
            "min_level": filter.min_level.map(|l| l.label()),
            "substr": filter.substr,
        },
        "tail": tail.iter().map(|l| json!({
            "ts_ms": l.ts_ms,
            "level": l.level.label(),
            "message": l.message,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("tui-log-tail.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_filter_drops_below_min() {
        let f = LogFilter {
            min_level: Some(Level::Warn),
            substr: None,
        };
        let lines = demo_log();
        let out = filter_lines(&lines, &f);
        assert!(out.iter().all(|l| l.level >= Level::Warn));
    }

    #[test]
    fn substr_filter_matches() {
        let f = LogFilter {
            min_level: None,
            substr: Some("timeout".into()),
        };
        let lines = demo_log();
        let out = filter_lines(&lines, &f);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn tail_n_returns_last_n() {
        let all = demo_log();
        let refs: Vec<&LogLine> = all.iter().collect();
        let t = tail_n(&refs, 3);
        assert_eq!(t.len(), 3);
        assert_eq!(t[2].message, all[9].message);
    }

    #[test]
    fn tail_n_clamps_to_len() {
        let all = demo_log();
        let refs: Vec<&LogLine> = all.iter().collect();
        let t = tail_n(&refs, 9999);
        assert_eq!(t.len(), all.len());
    }

    #[test]
    fn render_contains_header_and_footer() {
        let all = demo_log();
        let refs: Vec<&LogLine> = all.iter().take(2).collect();
        let s = render_tail(&refs);
        assert!(s.contains("LOG TAIL"));
        assert!(s.ends_with("-+\n"));
    }
}
