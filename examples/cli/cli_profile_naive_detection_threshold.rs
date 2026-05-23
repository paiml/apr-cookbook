//! # apr profile — `--detect-naive --threshold` Naive Implementation Gate
//!
//! `apr profile <FILE> --detect-naive --threshold <GFLOPS>` flags
//! operations whose measured throughput falls below `<GFLOPS>` of peak
//! roofline. The default 10.0 GFLOPS catches the most egregious naive
//! kernels (unrolled f32 matmul, bare-metal Python loops). This recipe
//! builds the verdict classifier as a pure function.
//!
//! Demonstrates the **PROFILE.4** recipe for PMAT-102 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROFILE-001 + Williams et al. (2009) Roofline model
//!
//! Run with: cargo run --example cli_profile_naive_detection_threshold
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct OpMeasurement {
    pub op: String,
    pub measured_gflops: f64,
    pub roofline_peak_gflops: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NaiveVerdict {
    Optimized { ratio: f64 },
    Naive { measured: f64, threshold: f64 },
    UnknownPeak,
}

pub fn classify(m: &OpMeasurement, threshold_gflops: f64) -> NaiveVerdict {
    if m.roofline_peak_gflops <= 0.0 || !m.roofline_peak_gflops.is_finite() {
        return NaiveVerdict::UnknownPeak;
    }
    if m.measured_gflops < threshold_gflops {
        return NaiveVerdict::Naive {
            measured: m.measured_gflops,
            threshold: threshold_gflops,
        };
    }
    NaiveVerdict::Optimized {
        ratio: m.measured_gflops / m.roofline_peak_gflops,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_naive_detection_threshold")?;

    let measurements = [
        OpMeasurement {
            op: "attention_qkv".into(),
            measured_gflops: 250.0,
            roofline_peak_gflops: 312.0,
        },
        OpMeasurement {
            op: "ffn_gate_proj".into(),
            measured_gflops: 8.5,
            roofline_peak_gflops: 312.0,
        },
        OpMeasurement {
            op: "naive_softmax".into(),
            measured_gflops: 0.4,
            roofline_peak_gflops: 312.0,
        },
    ];

    for t in [10.0_f64, 50.0, 200.0] {
        println!("--threshold {t} GFLOPS:");
        for m in &measurements {
            println!("  {}  →  {:?}", m.op, classify(m, t));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(name: &str, measured: f64, peak: f64) -> OpMeasurement {
        OpMeasurement {
            op: name.into(),
            measured_gflops: measured,
            roofline_peak_gflops: peak,
        }
    }

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn optimized_op_passes() {
        let m = make("ok", 250.0, 312.0);
        let v = classify(&m, 10.0);
        if let NaiveVerdict::Optimized { ratio } = v {
            assert!((ratio - 0.8013).abs() < 0.01);
        } else {
            panic!("expected Optimized");
        }
    }

    #[test]
    fn below_threshold_flagged_naive() {
        let m = make("slow", 5.0, 312.0);
        let v = classify(&m, 10.0);
        assert!(matches!(v, NaiveVerdict::Naive { .. }));
    }

    #[test]
    fn boundary_at_exact_threshold_passes_as_optimized() {
        // Conservative-pass at boundary (≥ threshold).
        let m = make("borderline", 10.0, 312.0);
        let v = classify(&m, 10.0);
        assert!(matches!(v, NaiveVerdict::Optimized { .. }));
    }

    #[test]
    fn zero_peak_returns_unknown_peak() {
        // Avoids divide-by-zero in ratio computation.
        let m = make("noop", 5.0, 0.0);
        assert_eq!(classify(&m, 10.0), NaiveVerdict::UnknownPeak);
    }

    #[test]
    fn negative_peak_returns_unknown_peak() {
        let m = make("garbage", 5.0, -1.0);
        assert_eq!(classify(&m, 10.0), NaiveVerdict::UnknownPeak);
    }

    #[test]
    fn nan_peak_returns_unknown_peak() {
        let m = make("nan", 5.0, f64::NAN);
        assert_eq!(classify(&m, 10.0), NaiveVerdict::UnknownPeak);
    }

    #[test]
    fn higher_threshold_catches_more_ops() {
        let m = make("medium", 50.0, 312.0);
        // 50 GFLOPS at threshold 10 → Optimized; at threshold 100 → Naive.
        assert!(matches!(classify(&m, 10.0), NaiveVerdict::Optimized { .. }));
        assert!(matches!(classify(&m, 100.0), NaiveVerdict::Naive { .. }));
    }
}
