//! # Distillation Teacher Ensemble Voting
//!
//! Combine multiple teacher predictions:
//!   Mean: average logits per class (smoothest)
//!   Majority: most common argmax (robust to outliers)
//!   Median: per-class median logit (rejects bad teachers)
//!
//! Demonstrates the **DIST.26** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015) ensemble distillation.
//!
//! Run with: cargo run --example distill_teacher_ensemble
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteStrategy {
    Mean,
    Majority,
    Median,
}

#[derive(Debug, PartialEq)]
pub enum EnsembleVerdict {
    Ok { combined: Vec<f64> },
    EmptyTeachers,
    DimensionMismatch,
}

pub fn combine(teacher_logits: &[Vec<f64>], strategy: VoteStrategy) -> EnsembleVerdict {
    if teacher_logits.is_empty() {
        return EnsembleVerdict::EmptyTeachers;
    }
    let dim = teacher_logits[0].len();
    if dim == 0 || teacher_logits.iter().any(|t| t.len() != dim) {
        return EnsembleVerdict::DimensionMismatch;
    }
    let combined = match strategy {
        VoteStrategy::Mean => {
            let n = teacher_logits.len() as f64;
            let mut out = vec![0.0; dim];
            for t in teacher_logits {
                for (i, v) in t.iter().enumerate() {
                    out[i] += v;
                }
            }
            for v in &mut out {
                *v /= n;
            }
            out
        }
        VoteStrategy::Majority => {
            let mut votes = vec![0u32; dim];
            for t in teacher_logits {
                let argmax = argmax(t);
                votes[argmax] += 1;
            }
            let winner = argmax_u32(&votes);
            let mut out = vec![0.0; dim];
            out[winner] = 1.0;
            out
        }
        VoteStrategy::Median => {
            let mut out = vec![0.0; dim];
            for c in 0..dim {
                let mut col: Vec<f64> = teacher_logits.iter().map(|t| t[c]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out[c] = col[col.len() / 2];
            }
            out
        }
    };
    EnsembleVerdict::Ok { combined }
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

fn argmax_u32(v: &[u32]) -> usize {
    v.iter()
        .enumerate()
        .max_by_key(|(_, x)| *x)
        .map_or(0, |(i, _)| i)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_teacher_ensemble")?;

    let teachers = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.5, 2.5, 2.5],
        vec![0.5, 3.0, 3.5],
    ];
    println!("mean: {:?}", combine(&teachers, VoteStrategy::Mean));
    println!("majority: {:?}", combine(&teachers, VoteStrategy::Majority));
    println!("median: {:?}", combine(&teachers, VoteStrategy::Median));
    println!("empty: {:?}", combine(&[], VoteStrategy::Mean));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 2.5],
            vec![0.5, 3.0, 3.5],
        ]
    }

    #[test]
    fn ensemble_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn mean_averages_correctly() {
        let v = combine(&typical(), VoteStrategy::Mean);
        if let EnsembleVerdict::Ok { combined } = v {
            // First class: (1.0 + 1.5 + 0.5) / 3 = 1.0.
            assert!((combined[0] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn majority_picks_dominant() {
        let v = combine(&typical(), VoteStrategy::Majority);
        if let EnsembleVerdict::Ok { combined } = v {
            // All three teachers' argmax is class 2.
            assert!((combined[2] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn median_rejects_outlier() {
        // Two teachers say class 1 = 2.0, one says class 1 = 100. Median = 2.0.
        let teachers = vec![vec![0.0, 2.0], vec![0.0, 2.0], vec![0.0, 100.0]];
        let v = combine(&teachers, VoteStrategy::Median);
        if let EnsembleVerdict::Ok { combined } = v {
            assert!((combined[1] - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_teachers_rejected() {
        assert_eq!(
            combine(&[], VoteStrategy::Mean),
            EnsembleVerdict::EmptyTeachers
        );
    }

    #[test]
    fn dim_mismatch_rejected() {
        let bad = vec![vec![1.0, 2.0], vec![1.0]];
        assert_eq!(
            combine(&bad, VoteStrategy::Mean),
            EnsembleVerdict::DimensionMismatch
        );
    }

    #[test]
    fn empty_dim_rejected() {
        let bad = vec![vec![]];
        assert_eq!(
            combine(&bad, VoteStrategy::Mean),
            EnsembleVerdict::DimensionMismatch
        );
    }

    #[test]
    fn single_teacher_is_passthrough() {
        let single = vec![vec![1.0, 2.0, 3.0]];
        if let EnsembleVerdict::Ok { combined } = combine(&single, VoteStrategy::Mean) {
            assert_eq!(combined, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn mean_smoother_than_majority() {
        // Mean spreads probability; majority is one-hot.
        let v_mean = combine(&typical(), VoteStrategy::Mean);
        let v_maj = combine(&typical(), VoteStrategy::Majority);
        if let (EnsembleVerdict::Ok { combined: m }, EnsembleVerdict::Ok { combined: j }) =
            (v_mean, v_maj)
        {
            // Mean has > 1 nonzero values; majority has 1.
            let mean_nonzero = m.iter().filter(|&&x| x > 0.01).count();
            let maj_nonzero = j.iter().filter(|&&x| x > 0.5).count();
            assert!(mean_nonzero > maj_nonzero);
        }
    }

    #[test]
    fn dimension_count_preserved() {
        let v = combine(&typical(), VoteStrategy::Mean);
        if let EnsembleVerdict::Ok { combined } = v {
            assert_eq!(combined.len(), 3);
        }
    }

    #[test]
    fn deterministic() {
        let a = combine(&typical(), VoteStrategy::Mean);
        let b = combine(&typical(), VoteStrategy::Mean);
        assert_eq!(a, b);
    }
}
