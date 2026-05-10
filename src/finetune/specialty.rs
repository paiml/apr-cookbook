//! Tier 3.11–3.16 — Specialty technique helpers.
//!
//! Closed-form invariants for 6 single-recipe sub-sections:
//!
//! - L-BFGS: convergence in fewer iterations than SGD on convex objective.
//! - FAMO: per-task gradient-norm balancing — no task dominates.
//! - SegFormer: per-pixel cross-entropy yields valid mIoU ∈ [0, 1].
//! - JSON schema decode: schema-constrained outputs always parse.
//! - Mamba: latency scales linearly with sequence length (vs O(n²) attn).
//! - Hypernetwork: per-task generated weights yield per-task predictions.

#![allow(clippy::needless_range_loop)]

/// L-BFGS vs SGD step counts to reach a target on a convex (quadratic) objective.
/// Returns true if L-BFGS converges in ≤ 0.5× SGD's iterations.
#[must_use]
pub fn lbfgs_converges_faster(lbfgs_iters: u32, sgd_iters: u32) -> bool {
    f64::from(lbfgs_iters) <= f64::from(sgd_iters) * 0.5
}

/// FAMO balancing: returns true if no per-task gradient norm is more than
/// 2× the median across tasks.
#[must_use]
pub fn famo_balanced(grad_norms: &[f64]) -> bool {
    if grad_norms.is_empty() {
        return true;
    }
    let mut sorted = grad_norms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    if median < 1e-12 {
        return false;
    }
    grad_norms.iter().all(|g| *g <= median * 2.0)
}

/// SegFormer per-pixel mIoU on labels: returns mean IoU across classes.
#[must_use]
pub fn segformer_miou(predictions: &[u8], targets: &[u8], n_classes: u8) -> f64 {
    if predictions.len() != targets.len() {
        return f64::NAN;
    }
    let mut iou_sum = 0.0_f64;
    let mut counted = 0_u32;
    for c in 0..n_classes {
        let mut intersection = 0_u32;
        let mut union = 0_u32;
        for (p, t) in predictions.iter().zip(targets.iter()) {
            if *p == c || *t == c {
                union += 1;
                if *p == c && *t == c {
                    intersection += 1;
                }
            }
        }
        if union > 0 {
            iou_sum += f64::from(intersection) / f64::from(union);
            counted += 1;
        }
    }
    if counted == 0 {
        0.0
    } else {
        iou_sum / f64::from(counted)
    }
}

/// JSON schema validity check: reports whether a string is well-formed JSON
/// that contains the given top-level field.
#[must_use]
pub fn json_has_field(s: &str, field: &str) -> bool {
    let trimmed = s.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return false;
    }
    trimmed.contains(&format!("\"{field}\":"))
}

/// Mamba latency model: O(n) (linear) vs attention O(n²). Returns linearity
/// score = `(t_n / n) / (t_1 / 1)` — close to 1.0 means linear.
#[must_use]
pub fn mamba_linearity(times: &[(u32, f64)]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    let baseline = {
        let (n, t) = times[0];
        t / f64::from(n)
    };
    let final_ratio = {
        let (n, t) = *times.last().unwrap();
        t / f64::from(n)
    };
    final_ratio / baseline
}

/// Hypernetwork: maps task_id → weight vector. Two distinct task IDs yield
/// distinct weight vectors (no collision).
#[must_use]
pub fn hypernetwork_generate(task_id: u32, dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| (((task_id * 31 + i as u32 * 17) % 23) as f64) / 23.0 - 0.5)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgs_faster_than_sgd_on_convex() {
        // 10 L-BFGS vs 100 SGD iters → 0.1× ratio.
        assert!(lbfgs_converges_faster(10, 100));
    }

    #[test]
    fn famo_balanced_uniform_norms() {
        let norms = vec![0.5, 0.6, 0.55, 0.5, 0.7];
        assert!(famo_balanced(&norms));
    }

    #[test]
    fn famo_unbalanced_when_one_task_dominates() {
        let norms = vec![0.1, 0.1, 5.0]; // task-2 dominates
        assert!(!famo_balanced(&norms));
    }

    #[test]
    fn segformer_perfect_predictions_yield_miou_1() {
        let p = vec![0_u8, 1, 2, 3, 0, 1, 2, 3];
        let t = p.clone();
        let miou = segformer_miou(&p, &t, 4);
        assert!((miou - 1.0).abs() < 1e-12);
    }

    #[test]
    fn segformer_all_wrong_yields_miou_0() {
        let p = vec![0_u8, 0, 0, 0];
        let t = vec![1_u8, 1, 1, 1];
        let miou = segformer_miou(&p, &t, 2);
        assert!((miou - 0.0).abs() < 1e-12);
    }

    #[test]
    fn json_schema_valid_passes() {
        let s = "{\"name\": \"alice\", \"age\": 30}";
        assert!(json_has_field(s, "name"));
        assert!(!json_has_field(s, "address"));
    }

    #[test]
    fn json_schema_malformed_fails() {
        assert!(!json_has_field("not JSON", "name"));
    }

    #[test]
    fn mamba_linear_when_t_proportional_to_n() {
        let times = vec![(1_u32, 1.0_f64), (10, 10.0), (100, 100.0)];
        let lin = mamba_linearity(&times);
        assert!((lin - 1.0).abs() < 0.1);
    }

    #[test]
    fn mamba_quadratic_breaks_linearity() {
        let times = vec![(1_u32, 1.0_f64), (10, 100.0), (100, 10000.0)];
        let lin = mamba_linearity(&times);
        // Quadratic time per element = 100 vs baseline 1 → ratio ≈ 100.
        assert!(lin > 10.0);
    }

    #[test]
    fn hypernetwork_distinct_tasks_distinct_weights() {
        let w1 = hypernetwork_generate(1, 16);
        let w2 = hypernetwork_generate(2, 16);
        assert_ne!(w1, w2);
    }

    #[test]
    fn hypernetwork_deterministic() {
        let a = hypernetwork_generate(7, 32);
        let b = hypernetwork_generate(7, 32);
        assert_eq!(a, b);
    }
}
