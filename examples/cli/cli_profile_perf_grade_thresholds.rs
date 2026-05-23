//! # apr profile — `--perf-grade` Letter-Grade Thresholds
//!
//! `apr profile <FILE> --perf-grade` computes a letter grade (A/B/C/D/F)
//! based on the throughput ratio vs an Ollama baseline. Thresholds are
//! deterministic so the grade change in CI is meaningful. This recipe
//! builds the grader and asserts the bucket boundaries.
//!
//! Demonstrates the **PROFILE.6** recipe for PMAT-102 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROFILE-003 + Ollama baseline convention
//!
//! Run with: cargo run --example cli_profile_perf_grade_thresholds
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerfGrade {
    F,
    D,
    C,
    B,
    A,
}

pub fn grade(speedup: f64) -> Option<PerfGrade> {
    if !speedup.is_finite() || speedup < 0.0 {
        return None;
    }
    Some(match speedup {
        s if s >= 2.0 => PerfGrade::A,
        s if s >= 1.5 => PerfGrade::B,
        s if s >= 1.0 => PerfGrade::C,
        s if s >= 0.5 => PerfGrade::D,
        _ => PerfGrade::F,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_perf_grade_thresholds")?;

    for s in [
        0.1_f64,
        0.5,
        0.9,
        1.0,
        1.4,
        1.5,
        1.9,
        2.0,
        5.0,
        -0.1,
        f64::NAN,
    ] {
        println!("speedup {s:>6.2}x  →  {:?}", grade(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn boundary_at_2x_is_a_grade() {
        assert_eq!(grade(2.0), Some(PerfGrade::A));
        assert_eq!(grade(1.99), Some(PerfGrade::B));
    }

    #[test]
    fn boundary_at_1x_is_c_grade() {
        // 1x = parity with baseline; an honest C, not a B.
        assert_eq!(grade(1.0), Some(PerfGrade::C));
    }

    #[test]
    fn below_half_is_f_grade() {
        assert_eq!(grade(0.4), Some(PerfGrade::F));
        assert_eq!(grade(0.0), Some(PerfGrade::F));
    }

    #[test]
    fn above_2x_still_a_grade() {
        // No A+ — just A.
        assert_eq!(grade(5.0), Some(PerfGrade::A));
        assert_eq!(grade(100.0), Some(PerfGrade::A));
    }

    #[test]
    fn negative_speedup_returns_none() {
        // Negative speedup is nonsense — return None rather than F.
        assert!(grade(-0.1).is_none());
    }

    #[test]
    fn nan_speedup_returns_none() {
        assert!(grade(f64::NAN).is_none());
    }

    #[test]
    fn inf_speedup_returns_none() {
        assert!(grade(f64::INFINITY).is_none());
    }

    #[test]
    fn grade_ordering_matches_alphabetic_intuition() {
        assert!(PerfGrade::F < PerfGrade::D);
        assert!(PerfGrade::D < PerfGrade::C);
        assert!(PerfGrade::C < PerfGrade::B);
        assert!(PerfGrade::B < PerfGrade::A);
    }
}
