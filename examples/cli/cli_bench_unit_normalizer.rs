//! # apr bench --unit — Output Unit Normalizer
//!
//! `apr bench` reports latency in seconds, milliseconds, microseconds,
//! or nanoseconds. Auto-pick rule: smallest unit where mean > 1.0
//! avoids false-precision (e.g., "0.000001 s" → "1.0 µs"). This recipe
//! builds the normalizer.
//!
//! Demonstrates the **BENCH.6** recipe for PMAT-118 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender BENCH-001 + criterion.rs unit conventions
//!
//! Run with: cargo run --example cli_bench_unit_normalizer
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimeUnit {
    pub fn label(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "s",
            TimeUnit::Milliseconds => "ms",
            TimeUnit::Microseconds => "µs",
            TimeUnit::Nanoseconds => "ns",
        }
    }

    pub fn factor_from_seconds(self) -> f64 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Milliseconds => 1_000.0,
            TimeUnit::Microseconds => 1_000_000.0,
            TimeUnit::Nanoseconds => 1_000_000_000.0,
        }
    }
}

pub fn pick_unit(seconds: f64) -> Option<TimeUnit> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    if seconds >= 1.0 {
        Some(TimeUnit::Seconds)
    } else if seconds >= 1e-3 {
        Some(TimeUnit::Milliseconds)
    } else if seconds >= 1e-6 {
        Some(TimeUnit::Microseconds)
    } else {
        Some(TimeUnit::Nanoseconds)
    }
}

pub fn render(seconds: f64) -> Option<String> {
    let unit = pick_unit(seconds)?;
    let value = seconds * unit.factor_from_seconds();
    Some(format!("{:.3} {}", value, unit.label()))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_unit_normalizer")?;

    for s in [120.0, 1.5, 0.025, 4.5e-5, 7.2e-8, -1.0, f64::NAN] {
        println!("{s:>12} s  →  {:?}", render(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn over_one_second_uses_seconds() {
        assert_eq!(pick_unit(2.5), Some(TimeUnit::Seconds));
    }

    #[test]
    fn millisecond_range_uses_ms() {
        assert_eq!(pick_unit(0.005), Some(TimeUnit::Milliseconds));
    }

    #[test]
    fn microsecond_range_uses_us() {
        assert_eq!(pick_unit(50e-6), Some(TimeUnit::Microseconds));
    }

    #[test]
    fn nanosecond_range_uses_ns() {
        assert_eq!(pick_unit(50e-9), Some(TimeUnit::Nanoseconds));
    }

    #[test]
    fn negative_or_nan_rejected() {
        assert!(pick_unit(-1.0).is_none());
        assert!(pick_unit(f64::NAN).is_none());
    }

    #[test]
    fn boundary_at_one_second_uses_seconds() {
        assert_eq!(pick_unit(1.0), Some(TimeUnit::Seconds));
    }

    #[test]
    fn render_formats_with_label() {
        let s = render(0.0025).unwrap();
        assert!(s.ends_with(" ms"));
        assert!(s.contains("2.5"));
    }

    #[test]
    fn render_microseconds() {
        // 4.5e-5 s = 45 µs.
        let s = render(4.5e-5).unwrap();
        assert!(s.contains("µs"));
        assert!(s.contains("45"));
    }

    #[test]
    fn factor_round_trip() {
        // value × factor / factor returns original seconds.
        for s in [1.0, 0.001, 1e-6, 1e-9] {
            let unit = pick_unit(s).unwrap();
            let scaled = s * unit.factor_from_seconds();
            let back = scaled / unit.factor_from_seconds();
            assert!((back - s).abs() < 1e-15);
        }
    }
}
