//! # Recipe: Eval Benchmark Suite (HellaSwag / ARC / MMLU-style)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr eval model.apr --suite hellaswag,arc,mmlu --format json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example eval_benchmark_suite` exits 0
//! 2. [x] `cargo test --example eval_benchmark_suite` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr eval --suite` in-process (no shell-out)
//! 10. [x] Unit tests cover accuracy, empty suite, macro averaging
//!
//! ## Learning Objective
//! Demonstrates a benchmark-suite runner that evaluates accuracy across three
//! synthetic multiple-choice tasks (HellaSwag-style commonsense, ARC-style
//! science, MMLU-style domain) and reports per-task and macro-averaged scores.
//!
//! ## Run Command
//! ```bash
//! cargo run --example eval_benchmark_suite
//! ```
//!
//! ## References
//! - Liang, P. et al. (2023). *HELM: Holistic Evaluation of Language Models*. TMLR. arXiv:2211.09110

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone)]
struct MultipleChoiceItem {
    _prompt: String,
    choices: Vec<String>,
    correct_idx: usize,
}

#[derive(Debug, Clone)]
struct TaskResult {
    task: String,
    n_items: usize,
    n_correct: usize,
    accuracy: f64,
}

impl TaskResult {
    fn verdict(&self, threshold: f64) -> &'static str {
        if self.accuracy >= threshold {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

fn synthesize_task(
    rng: &mut impl Rng,
    task_name: &str,
    n_items: usize,
    n_choices: usize,
) -> Vec<MultipleChoiceItem> {
    (0..n_items)
        .map(|i| {
            let correct_idx = rng.gen_range(0..n_choices);
            let choices: Vec<String> = (0..n_choices)
                .map(|c| format!("{}-opt-{}", task_name, c))
                .collect();
            MultipleChoiceItem {
                _prompt: format!("{}-item-{}", task_name, i),
                choices,
                correct_idx,
            }
        })
        .collect()
}

fn predict_item(rng: &mut impl Rng, item: &MultipleChoiceItem, skill: f64) -> usize {
    // With probability `skill`, pick the correct answer. Otherwise uniform.
    if rng.gen_bool(skill) {
        item.correct_idx
    } else {
        rng.gen_range(0..item.choices.len())
    }
}

fn evaluate_task(
    rng: &mut impl Rng,
    task: &str,
    items: &[MultipleChoiceItem],
    skill: f64,
) -> TaskResult {
    let n_correct = items
        .iter()
        .filter(|it| predict_item(rng, it, skill) == it.correct_idx)
        .count();
    let n_items = items.len();
    let accuracy = if n_items == 0 {
        0.0
    } else {
        n_correct as f64 / n_items as f64
    };
    TaskResult {
        task: task.to_string(),
        n_items,
        n_correct,
        accuracy,
    }
}

fn macro_average(results: &[TaskResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.accuracy).sum::<f64>() / results.len() as f64
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("eval_benchmark_suite")?;
    println!("=== Recipe: {} ===", ctx.name());

    let threshold = 0.55;
    let tasks = [
        ("hellaswag", 80, 4, 0.75),
        ("arc", 60, 4, 0.65),
        ("mmlu", 100, 4, 0.55),
    ];

    let mut results = Vec::new();
    for (name, n, k, skill) in tasks {
        let items = synthesize_task(ctx.rng(), name, n, k);
        let r = evaluate_task(ctx.rng(), name, &items, skill);
        println!(
            "{:<12} items={:>4} correct={:>4} accuracy={:.4} [{}]",
            r.task,
            r.n_items,
            r.n_correct,
            r.accuracy,
            r.verdict(threshold),
        );
        results.push(r);
    }

    let macro_acc = macro_average(&results);
    println!("\nMacro-average accuracy: {:.4}", macro_acc);
    println!(
        "Overall verdict: {}",
        if macro_acc >= threshold {
            "PASS"
        } else {
            "FAIL"
        }
    );

    let report = json!({
        "recipe": ctx.name(),
        "threshold": threshold,
        "tasks": results.iter().map(|r| json!({
            "task": r.task,
            "n_items": r.n_items,
            "n_correct": r.n_correct,
            "accuracy": r.accuracy,
        })).collect::<Vec<_>>(),
        "macro_accuracy": macro_acc,
    });
    let out_path = ctx.path("eval-suite.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, bytes)?;

    ctx.record_float_metric("macro_accuracy", macro_acc);
    ctx.record_metric("n_tasks", results.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn item(correct: usize) -> MultipleChoiceItem {
        MultipleChoiceItem {
            _prompt: String::new(),
            choices: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            correct_idx: correct,
        }
    }

    #[test]
    fn perfect_skill_yields_full_accuracy() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = vec![item(0), item(1), item(2)];
        let r = evaluate_task(&mut rng, "t", &items, 1.0);
        assert_eq!(r.n_correct, 3);
        assert!((r.accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_skill_still_produces_nonzero_chance() {
        let mut rng = StdRng::seed_from_u64(7);
        let items: Vec<_> = (0..400).map(|i| item(i % 4)).collect();
        let r = evaluate_task(&mut rng, "t", &items, 0.0);
        // Uniform guessing over 4 choices => ~25% accuracy.
        assert!(r.accuracy > 0.15 && r.accuracy < 0.35);
    }

    #[test]
    fn macro_average_of_empty_is_zero() {
        assert_eq!(macro_average(&[]), 0.0);
    }

    #[test]
    fn macro_average_of_two_tasks() {
        let a = TaskResult {
            task: "a".into(),
            n_items: 10,
            n_correct: 8,
            accuracy: 0.8,
        };
        let b = TaskResult {
            task: "b".into(),
            n_items: 10,
            n_correct: 6,
            accuracy: 0.6,
        };
        assert!((macro_average(&[a, b]) - 0.7).abs() < 1e-9);
    }
}
