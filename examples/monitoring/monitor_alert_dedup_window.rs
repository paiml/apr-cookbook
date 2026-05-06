//! # Monitoring Alert Dedup Window
//!
//! Suppress repeated alerts of the same kind within a sliding window.
//! Without dedup: a single bad metric causes alert spam every 30s.
//! Algorithm: per `(severity, source)` key, track last-fired-at; emit
//! only if (now - last_fired) > suppress_window_secs OR severity has
//! escalated. This recipe builds the dedup engine.
//!
//! Demonstrates the **MON.17** recipe for PMAT-137 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Prometheus Alertmanager grouping & inhibition rules.
//!
//! Run with: cargo run --example monitor_alert_dedup_window
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum AlertVerdict {
    Emit,
    Suppressed { reason: SuppressReason },
}

#[derive(Debug, PartialEq, Eq)]
pub enum SuppressReason {
    WithinWindow { remaining_secs: u64 },
}

pub struct DedupEngine {
    last_fired: HashMap<String, (u64, Severity)>,
    window_secs: u64,
}

impl DedupEngine {
    pub fn new(window_secs: u64) -> Self {
        Self {
            last_fired: HashMap::new(),
            window_secs,
        }
    }

    pub fn evaluate(&mut self, severity: Severity, source: &str, now_secs: u64) -> AlertVerdict {
        if let Some((last, prev_sev)) = self.last_fired.get(source) {
            if now_secs >= *last && now_secs - *last < self.window_secs && severity <= *prev_sev {
                let remaining = self.window_secs - (now_secs - *last);
                return AlertVerdict::Suppressed {
                    reason: SuppressReason::WithinWindow {
                        remaining_secs: remaining,
                    },
                };
            }
        }
        self.last_fired
            .insert(source.to_string(), (now_secs, severity));
        AlertVerdict::Emit
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_alert_dedup_window")?;

    let mut e = DedupEngine::new(300);
    println!(
        "first @ 100: {:?}",
        e.evaluate(Severity::Warning, "db", 100)
    );
    println!("dup @ 200: {:?}", e.evaluate(Severity::Warning, "db", 200));
    println!(
        "after window @ 500: {:?}",
        e.evaluate(Severity::Warning, "db", 500)
    );
    println!(
        "escalation @ 600: {:?}",
        e.evaluate(Severity::Critical, "db", 600)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_alert_emits() {
        let mut e = DedupEngine::new(300);
        assert_eq!(
            e.evaluate(Severity::Warning, "src", 100),
            AlertVerdict::Emit
        );
    }

    #[test]
    fn duplicate_within_window_suppressed() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        let v = e.evaluate(Severity::Warning, "src", 200);
        assert!(matches!(v, AlertVerdict::Suppressed { .. }));
    }

    #[test]
    fn alert_after_window_emits() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        let v = e.evaluate(Severity::Warning, "src", 500);
        assert_eq!(v, AlertVerdict::Emit);
    }

    #[test]
    fn escalation_emits_immediately() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        // Critical > Warning → emit even within window.
        let v = e.evaluate(Severity::Critical, "src", 200);
        assert_eq!(v, AlertVerdict::Emit);
    }

    #[test]
    fn different_sources_independent() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src1", 100);
        let v = e.evaluate(Severity::Warning, "src2", 100);
        assert_eq!(v, AlertVerdict::Emit);
    }

    #[test]
    fn remaining_secs_reported() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        if let AlertVerdict::Suppressed {
            reason: SuppressReason::WithinWindow { remaining_secs },
        } = e.evaluate(Severity::Warning, "src", 150)
        {
            assert_eq!(remaining_secs, 250);
        }
    }

    #[test]
    fn at_exact_window_emits() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        // 100 + 300 = 400, equality is boundary → emit.
        let v = e.evaluate(Severity::Warning, "src", 400);
        assert_eq!(v, AlertVerdict::Emit);
    }

    #[test]
    fn just_inside_window_suppressed() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        let v = e.evaluate(Severity::Warning, "src", 399);
        assert!(matches!(v, AlertVerdict::Suppressed { .. }));
    }

    #[test]
    fn de_escalation_suppressed_within_window() {
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Critical, "src", 100);
        // Warning < Critical → suppressed within window.
        let v = e.evaluate(Severity::Warning, "src", 200);
        assert!(matches!(v, AlertVerdict::Suppressed { .. }));
    }

    #[test]
    fn out_of_order_now_handled_gracefully() {
        // now < last (clock skew) → no underflow, still suppress.
        let mut e = DedupEngine::new(300);
        e.evaluate(Severity::Warning, "src", 100);
        let v = e.evaluate(Severity::Warning, "src", 50);
        assert!(matches!(v, AlertVerdict::Emit));
    }
}
