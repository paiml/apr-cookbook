//! # Recipe: Eval Pass@k for Code Generation
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr eval model.apr --dataset humaneval.jsonl --pass-at-k 1,5,10`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example eval_pass_at_k` exits 0
//! 2. [x] `cargo test --example eval_pass_at_k` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr eval --pass-at-k` in-process (no shell-out)
//! 10. [x] Unit tests cover HumanEval unbiased estimator edge cases
//!
//! ## Learning Objective
//! Computes pass@k using the unbiased estimator from HumanEval. For each problem
//! with n samples and c correct, pass@k = 1 - C(n-c, k)/C(n, k). Aggregates
//! across problems for pass@1, pass@5, pass@10.
//!
//! ## Run Command
//! ```bash
//! cargo run --example eval_pass_at_k
//! ```
//!
//! ## References
//! - Chen, M. et al. (2021). *Evaluating LLMs Trained on Code*. arXiv:2107.03374

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone)]
struct ProblemSamples {
    problem_id: String,
    n: usize,
    c: usize,
}

/// Unbiased pass@k estimator from Chen et al. 2021.
///
/// pass@k = 1 - C(n-c, k) / C(n, k), evaluated via the numerically-stable
/// product form 1 - Π_{i=0..k} (n - c - i) / (n - i).
fn pass_at_k_one(n: usize, c: usize, k: usize) -> f64 {
    if k == 0 || n == 0 {
        return 0.0;
    }
    if c == 0 {
        return 0.0;
    }
    if n - c < k {
        return 1.0;
    }
    let k = k.min(n);
    let mut prod = 1.0_f64;
    for i in 0..k {
        let num = (n - c - i) as f64;
        let den = (n - i) as f64;
        prod *= num / den;
    }
    1.0 - prod
}

fn aggregate_pass_at_k(problems: &[ProblemSamples], k: usize) -> f64 {
    if problems.is_empty() {
        return 0.0;
    }
    let sum: f64 = problems.iter().map(|p| pass_at_k_one(p.n, p.c, k)).sum();
    sum / problems.len() as f64
}

fn synthesize_problems(
    rng: &mut impl Rng,
    n_problems: usize,
    n_samples: usize,
) -> Vec<ProblemSamples> {
    (0..n_problems)
        .map(|i| {
            // Per-problem difficulty -> per-sample correctness probability.
            let difficulty = rng.gen_range(0.0_f64..1.0);
            let p_correct = (1.0 - difficulty).powf(1.5);
            let c = (0..n_samples).filter(|_| rng.gen_bool(p_correct)).count();
            ProblemSamples {
                problem_id: format!("P{:03}", i),
                n: n_samples,
                c,
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("eval_pass_at_k")?;
    println!("=== Recipe: {} ===", ctx.name());

    let n_problems = 30;
    let n_samples = 20;
    let ks = [1_usize, 5, 10];

    let problems = synthesize_problems(ctx.rng(), n_problems, n_samples);

    println!("\n--- Problem Distribution ---");
    println!("{:<8} {:>4} {:>4} {:>10}", "Problem", "n", "c", "c/n");
    for p in problems.iter().take(10) {
        println!(
            "{:<8} {:>4} {:>4} {:>10.3}",
            p.problem_id,
            p.n,
            p.c,
            p.c as f64 / p.n as f64
        );
    }
    println!("(showing 10 of {})", n_problems);

    println!("\n--- Pass@k Aggregate ---");
    let mut scores = Vec::new();
    for &k in &ks {
        let score = aggregate_pass_at_k(&problems, k);
        println!("pass@{:<3} = {:.4}", k, score);
        scores.push((k, score));
    }

    let report = json!({
        "recipe": ctx.name(),
        "n_problems": n_problems,
        "n_samples": n_samples,
        "scores": scores.iter().map(|(k, v)| json!({
            "k": k,
            "pass_at_k": v,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("pass-at-k.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    if let Some((_, top)) = scores.last() {
        ctx.record_float_metric("pass_at_10", *top);
    }
    ctx.record_metric("n_problems", n_problems as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_correct_gives_zero() {
        assert_eq!(pass_at_k_one(10, 0, 1), 0.0);
        assert_eq!(pass_at_k_one(10, 0, 5), 0.0);
    }

    #[test]
    fn all_correct_gives_one() {
        assert!((pass_at_k_one(10, 10, 1) - 1.0).abs() < 1e-12);
        assert!((pass_at_k_one(10, 10, 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn k_larger_than_failures_gives_one() {
        // n=5, c=3, k=3 => n-c=2 < k => pass@3 = 1.0
        assert!((pass_at_k_one(5, 3, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pass_at_1_equals_c_over_n() {
        // pass@1 = 1 - (n-c)/n = c/n
        let v = pass_at_k_one(20, 5, 1);
        assert!((v - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pass_at_k_monotone_in_k() {
        let a = pass_at_k_one(20, 3, 1);
        let b = pass_at_k_one(20, 3, 5);
        let c = pass_at_k_one(20, 3, 10);
        assert!(a <= b);
        assert!(b <= c);
    }

    #[test]
    fn aggregate_empty_is_zero() {
        assert_eq!(aggregate_pass_at_k(&[], 1), 0.0);
    }
}
