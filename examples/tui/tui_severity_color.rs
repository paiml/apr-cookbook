//! # TUI Severity → Color Mapper
//!
//! Map a severity level to ANSI color code for fg + bg. Levels:
//! Trace/Debug (gray), Info (cyan), Warn (yellow), Error (red),
//! Fatal (white-on-red).
//!
//! Demonstrates the **TUI.06** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ANSI escape codes (ECMA-48) + log4j level conventions.
//!
//! Run with: cargo run --example tui_severity_color
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

#[derive(Debug, PartialEq)]
pub enum ColorVerdict {
    Pick {
        fg_ansi: u32,
        bg_ansi: u32,
        bold: bool,
    },
}

pub fn pick(severity: Severity) -> ColorVerdict {
    let (fg, bg, bold) = match severity {
        Severity::Trace => (90, 49, false), // dark gray, default bg
        Severity::Debug => (37, 49, false), // light gray
        Severity::Info => (36, 49, false),  // cyan
        Severity::Warn => (33, 49, true),   // yellow + bold
        Severity::Error => (31, 49, true),  // red + bold
        Severity::Fatal => (37, 41, true),  // white on red + bold
    };
    ColorVerdict::Pick {
        fg_ansi: fg,
        bg_ansi: bg,
        bold,
    }
}

pub fn parse(level_str: &str) -> Option<Severity> {
    match level_str.trim().to_ascii_lowercase().as_str() {
        "trace" => Some(Severity::Trace),
        "debug" => Some(Severity::Debug),
        "info" => Some(Severity::Info),
        "warn" | "warning" => Some(Severity::Warn),
        "error" | "err" => Some(Severity::Error),
        "fatal" | "critical" | "crit" => Some(Severity::Fatal),
        _ => None,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_severity_color")?;

    println!("info: {:?}", pick(Severity::Info));
    println!("warn: {:?}", pick(Severity::Warn));
    println!("fatal: {:?}", pick(Severity::Fatal));
    println!("parse 'WARN': {:?}", parse("WARN"));
    println!("parse 'unknown': {:?}", parse("unknown"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn info_cyan_not_bold() {
        let v = pick(Severity::Info);
        if let ColorVerdict::Pick { fg_ansi, bold, .. } = v {
            assert_eq!(fg_ansi, 36);
            assert!(!bold);
        }
    }

    #[test]
    fn warn_yellow_bold() {
        let v = pick(Severity::Warn);
        if let ColorVerdict::Pick { fg_ansi, bold, .. } = v {
            assert_eq!(fg_ansi, 33);
            assert!(bold);
        }
    }

    #[test]
    fn error_red_bold() {
        let v = pick(Severity::Error);
        if let ColorVerdict::Pick { fg_ansi, bold, .. } = v {
            assert_eq!(fg_ansi, 31);
            assert!(bold);
        }
    }

    #[test]
    fn fatal_white_on_red_bold() {
        let v = pick(Severity::Fatal);
        if let ColorVerdict::Pick {
            fg_ansi,
            bg_ansi,
            bold,
        } = v
        {
            assert_eq!(fg_ansi, 37);
            assert_eq!(bg_ansi, 41);
            assert!(bold);
        }
    }

    #[test]
    fn parse_lowercase() {
        assert_eq!(parse("info"), Some(Severity::Info));
    }

    #[test]
    fn parse_uppercase() {
        assert_eq!(parse("INFO"), Some(Severity::Info));
    }

    #[test]
    fn parse_with_whitespace() {
        assert_eq!(parse("  warn  "), Some(Severity::Warn));
    }

    #[test]
    fn parse_alias() {
        assert_eq!(parse("warning"), Some(Severity::Warn));
        assert_eq!(parse("err"), Some(Severity::Error));
        assert_eq!(parse("critical"), Some(Severity::Fatal));
    }

    #[test]
    fn parse_unknown_none() {
        assert_eq!(parse("unknown"), None);
    }

    #[test]
    fn parse_empty_none() {
        assert_eq!(parse(""), None);
    }

    #[test]
    fn deterministic() {
        let a = pick(Severity::Warn);
        let b = pick(Severity::Warn);
        assert_eq!(a, b);
    }
}
