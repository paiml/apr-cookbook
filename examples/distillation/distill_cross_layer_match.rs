//! # Distillation Cross-Layer Attention Matching
//!
//! Match teacher layer L_t to student layer L_s by minimizing
//! cosine distance of attention patterns. Pair up layers greedy
//! (each teacher layer matched to its best unused student layer).
//!
//! Demonstrates the **DIST.25** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: BERT-PKD layer-mapping techniques (Sun et al., 2019).
//!
//! Run with: cargo run --example distill_cross_layer_match
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MatchVerdict {
    Ok {
        pairs: Vec<(usize, usize)>,
        avg_distance: f64,
    },
    EmptyTeacher,
    EmptyStudent,
    DimensionMismatch,
}

pub fn pair(teacher_layers: &[Vec<f64>], student_layers: &[Vec<f64>]) -> MatchVerdict {
    if teacher_layers.is_empty() {
        return MatchVerdict::EmptyTeacher;
    }
    if student_layers.is_empty() {
        return MatchVerdict::EmptyStudent;
    }
    let dim = teacher_layers[0].len();
    if dim == 0
        || teacher_layers.iter().any(|l| l.len() != dim)
        || student_layers.iter().any(|l| l.len() != dim)
    {
        return MatchVerdict::DimensionMismatch;
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut used: Vec<bool> = vec![false; student_layers.len()];
    let mut total = 0.0;
    let mut matched = 0;
    for (ti, t) in teacher_layers.iter().enumerate() {
        let mut best_idx: Option<usize> = None;
        let mut best_dist = f64::INFINITY;
        for (si, s) in student_layers.iter().enumerate() {
            if used[si] {
                continue;
            }
            let dist = cosine_distance(t, s);
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(si);
            }
        }
        if let Some(si) = best_idx {
            used[si] = true;
            pairs.push((ti, si));
            total += best_dist;
            matched += 1;
        }
    }
    let avg_distance = if matched > 0 {
        total / matched as f64
    } else {
        0.0
    };
    MatchVerdict::Ok {
        pairs,
        avg_distance,
    }
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_cross_layer_match")?;

    let teacher = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let student = vec![vec![0.0, 0.9, 0.1], vec![0.95, 0.05, 0.0]];
    println!("typical: {:?}", pair(&teacher, &student));
    println!("empty teacher: {:?}", pair(&[], &student));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_layers_zero_distance() {
        let l = vec![vec![1.0, 0.0, 0.0]];
        if let MatchVerdict::Ok { avg_distance, .. } = pair(&l, &l) {
            assert!(avg_distance.abs() < 1e-9);
        }
    }

    #[test]
    fn empty_teacher_rejected() {
        let s = vec![vec![1.0]];
        assert_eq!(pair(&[], &s), MatchVerdict::EmptyTeacher);
    }

    #[test]
    fn empty_student_rejected() {
        let t = vec![vec![1.0]];
        assert_eq!(pair(&t, &[]), MatchVerdict::EmptyStudent);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let t = vec![vec![1.0, 2.0]];
        let s = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(pair(&t, &s), MatchVerdict::DimensionMismatch);
    }

    #[test]
    fn each_teacher_paired_when_enough_students() {
        let t = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        if let MatchVerdict::Ok { pairs, .. } = pair(&t, &s) {
            assert_eq!(pairs.len(), 2);
        }
    }

    #[test]
    fn fewer_students_than_teacher_pairs_what_we_can() {
        let t = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let s = vec![vec![1.0, 0.0]];
        if let MatchVerdict::Ok { pairs, .. } = pair(&t, &s) {
            assert_eq!(pairs.len(), 1);
        }
    }

    #[test]
    fn pairs_sorted_by_teacher_index() {
        let t = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        if let MatchVerdict::Ok { pairs, .. } = pair(&t, &s) {
            assert_eq!(pairs[0].0, 0);
            assert_eq!(pairs[1].0, 1);
        }
    }

    #[test]
    fn perpendicular_high_distance() {
        let t = vec![vec![1.0, 0.0]];
        let s = vec![vec![0.0, 1.0]];
        if let MatchVerdict::Ok { avg_distance, .. } = pair(&t, &s) {
            assert!((avg_distance - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn opposite_max_distance() {
        let t = vec![vec![1.0, 0.0]];
        let s = vec![vec![-1.0, 0.0]];
        if let MatchVerdict::Ok { avg_distance, .. } = pair(&t, &s) {
            assert!((avg_distance - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn each_student_used_once() {
        let t = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let s = vec![vec![1.0, 0.0]];
        if let MatchVerdict::Ok { pairs, .. } = pair(&t, &s) {
            assert_eq!(pairs.len(), 1);
        }
    }

    #[test]
    fn empty_dim_rejected() {
        let t = vec![vec![]];
        let s = vec![vec![]];
        assert_eq!(pair(&t, &s), MatchVerdict::DimensionMismatch);
    }
}
