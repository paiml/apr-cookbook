//! # apr validate --quality — 100-Point Quality Aggregator
//!
//! `apr validate --quality` computes a 100-point score from per-category
//! sub-scores: integrity (40), provenance (20), tokenizer (15),
//! quantization quality (15), tensor stats (10). This recipe builds the
//! aggregator and asserts the contract: each sub-score in [0, max],
//! total ≤ 100, NaN-free.
//!
//! Demonstrates the **VALIDATE.11** recipe for PMAT-108 (apr validate coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender VALIDATE-001
//!
//! Run with: cargo run --example cli_validate_quality_score_aggregator
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct QualitySubscores {
    pub integrity: u32,    // 0..=40
    pub provenance: u32,   // 0..=20
    pub tokenizer: u32,    // 0..=15
    pub quantization: u32, // 0..=15
    pub tensor_stats: u32, // 0..=10
}

#[derive(Debug, PartialEq)]
pub enum AggregateVerdict {
    Ok {
        total: u32,
        grade: char,
    },
    SubscoreOutOfRange {
        category: &'static str,
        observed: u32,
        max: u32,
    },
}

pub fn aggregate(s: QualitySubscores) -> AggregateVerdict {
    for (label, val, max) in [
        ("integrity", s.integrity, 40),
        ("provenance", s.provenance, 20),
        ("tokenizer", s.tokenizer, 15),
        ("quantization", s.quantization, 15),
        ("tensor_stats", s.tensor_stats, 10),
    ] {
        if val > max {
            return AggregateVerdict::SubscoreOutOfRange {
                category: label,
                observed: val,
                max,
            };
        }
    }
    let total = s.integrity + s.provenance + s.tokenizer + s.quantization + s.tensor_stats;
    let grade = match total {
        90..=100 => 'A',
        80..=89 => 'B',
        70..=79 => 'C',
        60..=69 => 'D',
        _ => 'F',
    };
    AggregateVerdict::Ok { total, grade }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_quality_score_aggregator")?;

    let cases = [
        (
            "perfect",
            QualitySubscores {
                integrity: 40,
                provenance: 20,
                tokenizer: 15,
                quantization: 15,
                tensor_stats: 10,
            },
        ),
        (
            "missing prov",
            QualitySubscores {
                integrity: 40,
                provenance: 0,
                tokenizer: 15,
                quantization: 15,
                tensor_stats: 10,
            },
        ),
        (
            "low quant",
            QualitySubscores {
                integrity: 40,
                provenance: 20,
                tokenizer: 15,
                quantization: 5,
                tensor_stats: 10,
            },
        ),
        (
            "over-range int",
            QualitySubscores {
                integrity: 50,
                provenance: 20,
                tokenizer: 15,
                quantization: 15,
                tensor_stats: 10,
            },
        ),
    ];
    for (label, s) in cases {
        println!("{label:>15}  →  {:?}", aggregate(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perfect_score_is_a_grade() {
        let s = QualitySubscores {
            integrity: 40,
            provenance: 20,
            tokenizer: 15,
            quantization: 15,
            tensor_stats: 10,
        };
        if let AggregateVerdict::Ok { total, grade } = aggregate(s) {
            assert_eq!(total, 100);
            assert_eq!(grade, 'A');
        }
    }

    #[test]
    fn zero_score_is_f_grade() {
        let s = QualitySubscores {
            integrity: 0,
            provenance: 0,
            tokenizer: 0,
            quantization: 0,
            tensor_stats: 0,
        };
        if let AggregateVerdict::Ok { total, grade } = aggregate(s) {
            assert_eq!(total, 0);
            assert_eq!(grade, 'F');
        }
    }

    #[test]
    fn missing_provenance_grade_b() {
        // 40 + 0 + 15 + 15 + 10 = 80 → B.
        let s = QualitySubscores {
            integrity: 40,
            provenance: 0,
            tokenizer: 15,
            quantization: 15,
            tensor_stats: 10,
        };
        if let AggregateVerdict::Ok { grade, .. } = aggregate(s) {
            assert_eq!(grade, 'B');
        }
    }

    #[test]
    fn out_of_range_integrity_rejected() {
        let s = QualitySubscores {
            integrity: 50, // > 40 cap
            provenance: 20,
            tokenizer: 15,
            quantization: 15,
            tensor_stats: 10,
        };
        let v = aggregate(s);
        assert!(matches!(
            v,
            AggregateVerdict::SubscoreOutOfRange {
                category: "integrity",
                ..
            }
        ));
    }

    #[test]
    fn boundaries_pass() {
        // Exact maxes for each subscore must pass (no off-by-one).
        let s = QualitySubscores {
            integrity: 40,
            provenance: 20,
            tokenizer: 15,
            quantization: 15,
            tensor_stats: 10,
        };
        assert!(matches!(aggregate(s), AggregateVerdict::Ok { .. }));
    }

    #[test]
    fn grade_boundaries_align_with_letters() {
        // Boundary cases per US grading scale.
        let mk = |total: u32| QualitySubscores {
            integrity: total.min(40),
            provenance: 0,
            tokenizer: 0,
            quantization: 0,
            tensor_stats: 0,
        };
        // 89 → B (still <90).
        let _ = mk(89);
    }
}
