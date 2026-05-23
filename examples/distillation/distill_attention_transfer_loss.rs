//! # Distillation Attention-Transfer Loss
//!
//! Pass-through KL divergence on attention maps. For each layer:
//!   loss_l = KL(teacher_attn_l || student_attn_l)
//! Total loss = sum_l weight_l × loss_l.
//!
//! Use small weight (0.1-0.5) because attention loss is auxiliary;
//! main signal is still output-logit KD.
//!
//! Demonstrates the **DIST.14** recipe for PMAT-141 (distillation round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Zagoruyko & Komodakis (2017). Paying More Attention to Attention. arXiv:1612.03928.
//!
//! Run with: cargo run --example distill_attention_transfer_loss
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LossVerdict {
    Ok {
        total_loss: f64,
        per_layer: Vec<f64>,
    },
    LayerCountMismatch {
        teacher: usize,
        student: usize,
    },
    EmptyAttention,
    InvalidProbabilities,
    NegativeWeight,
}

pub fn compute(
    teacher_attns: &[Vec<f64>],
    student_attns: &[Vec<f64>],
    layer_weights: &[f64],
) -> LossVerdict {
    if teacher_attns.len() != student_attns.len() {
        return LossVerdict::LayerCountMismatch {
            teacher: teacher_attns.len(),
            student: student_attns.len(),
        };
    }
    if teacher_attns.is_empty() {
        return LossVerdict::EmptyAttention;
    }
    if layer_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return LossVerdict::NegativeWeight;
    }
    let mut per_layer = Vec::with_capacity(teacher_attns.len());
    for (l, (t, s)) in teacher_attns.iter().zip(student_attns.iter()).enumerate() {
        if t.len() != s.len() || t.is_empty() {
            return LossVerdict::InvalidProbabilities;
        }
        if !is_valid_distribution(t) || !is_valid_distribution(s) {
            return LossVerdict::InvalidProbabilities;
        }
        let kl = kl_divergence(t, s);
        let weight = layer_weights.get(l).copied().unwrap_or(1.0);
        per_layer.push(weight * kl);
    }
    let total_loss: f64 = per_layer.iter().sum();
    LossVerdict::Ok {
        total_loss,
        per_layer,
    }
}

fn is_valid_distribution(probs: &[f64]) -> bool {
    if probs.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return false;
    }
    let sum: f64 = probs.iter().sum();
    (sum - 1.0).abs() < 1e-3
}

fn kl_divergence(teacher: &[f64], student: &[f64]) -> f64 {
    teacher
        .iter()
        .zip(student.iter())
        .filter_map(|(t, s)| {
            if *t > 0.0 && *s > 0.0 {
                Some(t * (t / s).ln())
            } else {
                None
            }
        })
        .sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_attention_transfer_loss")?;

    let teacher = vec![vec![0.7, 0.2, 0.1], vec![0.5, 0.4, 0.1]];
    let student = vec![vec![0.6, 0.3, 0.1], vec![0.45, 0.45, 0.1]];
    println!("typical: {:?}", compute(&teacher, &student, &[0.5, 0.5]));

    let identical = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2]];
    println!(
        "identical: {:?}",
        compute(&identical, &identical, &[1.0, 1.0])
    );

    println!("empty: {:?}", compute(&[], &[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(p: &[f64]) -> Vec<f64> {
        p.to_vec()
    }

    #[test]
    fn loss_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_attentions_zero_loss() {
        let teacher = vec![dist(&[0.5, 0.3, 0.2])];
        let student = teacher.clone();
        if let LossVerdict::Ok { total_loss, .. } = compute(&teacher, &student, &[1.0]) {
            assert!(total_loss.abs() < 1e-9);
        }
    }

    #[test]
    fn divergent_attentions_positive_loss() {
        let teacher = vec![dist(&[0.7, 0.2, 0.1])];
        let student = vec![dist(&[0.1, 0.2, 0.7])];
        if let LossVerdict::Ok { total_loss, .. } = compute(&teacher, &student, &[1.0]) {
            assert!(total_loss > 0.0);
        }
    }

    #[test]
    fn layer_count_mismatch_rejected() {
        let teacher = vec![dist(&[0.5, 0.5])];
        let student = vec![dist(&[0.5, 0.5]), dist(&[0.4, 0.6])];
        let v = compute(&teacher, &student, &[1.0, 1.0]);
        assert!(matches!(v, LossVerdict::LayerCountMismatch { .. }));
    }

    #[test]
    fn empty_attention_rejected() {
        assert_eq!(compute(&[], &[], &[]), LossVerdict::EmptyAttention);
    }

    #[test]
    fn invalid_probability_rejected() {
        let teacher = vec![dist(&[1.5, -0.5])]; // not valid probs.
        let student = vec![dist(&[0.5, 0.5])];
        let v = compute(&teacher, &student, &[1.0]);
        assert_eq!(v, LossVerdict::InvalidProbabilities);
    }

    #[test]
    fn negative_weight_rejected() {
        let teacher = vec![dist(&[0.5, 0.5])];
        let student = vec![dist(&[0.5, 0.5])];
        let v = compute(&teacher, &student, &[-0.1]);
        assert_eq!(v, LossVerdict::NegativeWeight);
    }

    #[test]
    fn weighted_layers_sum_correctly() {
        let teacher = vec![dist(&[0.7, 0.3]), dist(&[0.6, 0.4])];
        let student = vec![dist(&[0.5, 0.5]), dist(&[0.5, 0.5])];
        let weights = [0.0, 1.0];
        if let LossVerdict::Ok {
            total_loss,
            per_layer,
        } = compute(&teacher, &student, &weights)
        {
            // Layer 0 weight 0 → contributes 0.
            assert!(per_layer[0].abs() < 1e-9);
            assert!((total_loss - per_layer[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn per_layer_loss_count_matches_input() {
        let teacher = vec![dist(&[0.5, 0.5]), dist(&[0.4, 0.6]), dist(&[0.3, 0.7])];
        let student = teacher.clone();
        if let LossVerdict::Ok { per_layer, .. } = compute(&teacher, &student, &[1.0, 1.0, 1.0]) {
            assert_eq!(per_layer.len(), 3);
        }
    }

    #[test]
    fn zero_weight_layer_zero_loss() {
        let teacher = vec![dist(&[0.7, 0.3])];
        let student = vec![dist(&[0.1, 0.9])];
        if let LossVerdict::Ok { total_loss, .. } = compute(&teacher, &student, &[0.0]) {
            assert!(total_loss.abs() < 1e-9);
        }
    }

    #[test]
    fn kl_zero_when_zero_in_teacher_skipped() {
        // KL(t||s) ignores positions where t == 0.
        let teacher = vec![dist(&[0.0, 0.5, 0.5])];
        let student = vec![dist(&[0.4, 0.3, 0.3])];
        let v = compute(&teacher, &student, &[1.0]);
        assert!(matches!(v, LossVerdict::Ok { .. }));
    }

    #[test]
    fn unequal_inner_lengths_rejected() {
        let teacher = vec![dist(&[0.5, 0.5])];
        let student = vec![dist(&[0.3, 0.3, 0.4])];
        let v = compute(&teacher, &student, &[1.0]);
        assert_eq!(v, LossVerdict::InvalidProbabilities);
    }
}
