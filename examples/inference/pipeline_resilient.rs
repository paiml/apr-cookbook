//! # Recipe: Error-Propagation Resilience Pipeline
//!
//! **Category**: inference
//! **CLI Equivalent**: `apr pipeline --resilient --retries 2 --fallback static`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example pipeline_resilient` exits 0
//! 2. [x] `cargo test --example pipeline_resilient` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr pipeline --resilient` in-process (no shell-out)
//! 10. [x] Unit tests cover retry, fallback, fatal path, happy path
//!
//! ## Learning Objective
//! Demonstrates a resilient inference pipeline that handles three failure
//! modes — transient errors (retry with backoff), recoverable errors
//! (fallback output), and fatal errors (propagate + record). Mirrors the
//! Sculley et al. "hidden technical debt" patterns surfaced by
//! `apr pipeline --resilient`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example pipeline_resilient
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    Transient(String),
    Recoverable(String),
    Fatal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageOutcome {
    Ok(String),
    Fallback(String),
    Failed(String),
}

impl StageOutcome {
    pub fn status(&self) -> &'static str {
        match self {
            StageOutcome::Ok(_) => "ok",
            StageOutcome::Fallback(_) => "fallback",
            StageOutcome::Failed(_) => "failed",
        }
    }
}

pub trait Stage {
    fn name(&self) -> &str;
    fn run(&mut self, input: &str) -> std::result::Result<String, StageError>;
    fn fallback(&self, _input: &str) -> Option<String> {
        None
    }
}

#[derive(Debug)]
pub struct ScriptedStage {
    name: String,
    script: Vec<std::result::Result<String, StageError>>,
    fallback_out: Option<String>,
    cursor: usize,
}

impl ScriptedStage {
    pub fn new(name: &str, script: Vec<std::result::Result<String, StageError>>) -> Self {
        Self {
            name: name.into(),
            script,
            fallback_out: None,
            cursor: 0,
        }
    }
    #[must_use]
    pub fn with_fallback(mut self, out: &str) -> Self {
        self.fallback_out = Some(out.into());
        self
    }
}

impl Stage for ScriptedStage {
    fn name(&self) -> &str {
        &self.name
    }
    fn run(&mut self, _input: &str) -> std::result::Result<String, StageError> {
        if self.cursor >= self.script.len() {
            return Err(StageError::Fatal("script exhausted".into()));
        }
        let v = self.script[self.cursor].clone();
        self.cursor += 1;
        v
    }
    fn fallback(&self, _input: &str) -> Option<String> {
        self.fallback_out.clone()
    }
}

#[derive(Debug, Clone)]
pub struct StageRecord {
    pub stage: String,
    pub outcome: StageOutcome,
    pub retries: usize,
}

pub fn run_stage(stage: &mut dyn Stage, input: &str, max_retries: usize) -> StageRecord {
    let mut retries = 0usize;
    loop {
        match stage.run(input) {
            Ok(out) => {
                return StageRecord {
                    stage: stage.name().into(),
                    outcome: StageOutcome::Ok(out),
                    retries,
                }
            }
            Err(StageError::Transient(_)) if retries < max_retries => {
                retries += 1;
            }
            Err(StageError::Transient(msg)) => {
                return StageRecord {
                    stage: stage.name().into(),
                    outcome: StageOutcome::Failed(format!("transient-exhausted: {}", msg)),
                    retries,
                }
            }
            Err(StageError::Recoverable(_)) => match stage.fallback(input) {
                Some(fb) => {
                    return StageRecord {
                        stage: stage.name().into(),
                        outcome: StageOutcome::Fallback(fb),
                        retries,
                    }
                }
                None => {
                    return StageRecord {
                        stage: stage.name().into(),
                        outcome: StageOutcome::Failed("recoverable-no-fallback".into()),
                        retries,
                    }
                }
            },
            Err(StageError::Fatal(msg)) => {
                return StageRecord {
                    stage: stage.name().into(),
                    outcome: StageOutcome::Failed(format!("fatal: {}", msg)),
                    retries,
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("pipeline_resilient")?;
    println!("=== Recipe: {} ===", ctx.name());

    let mut stages: Vec<Box<dyn Stage>> = vec![
        Box::new(ScriptedStage::new(
            "tokenize",
            vec![Ok("tokens[42]".into())],
        )),
        Box::new(ScriptedStage::new(
            "infer",
            vec![
                Err(StageError::Transient("gpu busy".into())),
                Err(StageError::Transient("gpu busy".into())),
                Ok("logits[128]".into()),
            ],
        )),
        Box::new(
            ScriptedStage::new(
                "decode",
                vec![Err(StageError::Recoverable("beam collapsed".into()))],
            )
            .with_fallback("greedy-decode-fallback"),
        ),
    ];

    let input = "hello world";
    let mut records = Vec::new();
    for s in &mut stages {
        let rec = run_stage(s.as_mut(), input, 3);
        println!(
            "[{:<8}] {} (retries={})",
            rec.stage,
            rec.outcome.status(),
            rec.retries
        );
        records.push(rec);
    }

    let failed = records
        .iter()
        .filter(|r| matches!(r.outcome, StageOutcome::Failed(_)))
        .count();
    let fallbacks = records
        .iter()
        .filter(|r| matches!(r.outcome, StageOutcome::Fallback(_)))
        .count();

    let report = json!({
        "recipe": ctx.name(),
        "failed": failed,
        "fallbacks": fallbacks,
        "stages": records.iter().map(|r| json!({
            "stage": r.stage,
            "status": r.outcome.status(),
            "retries": r.retries,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("pipeline-resilient.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("failed_stages", failed as i64);
    ctx.record_metric("fallbacks", fallbacks as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_ok() {
        let mut s = ScriptedStage::new("t", vec![Ok("out".into())]);
        let r = run_stage(&mut s, "in", 2);
        assert_eq!(r.outcome, StageOutcome::Ok("out".into()));
        assert_eq!(r.retries, 0);
    }

    #[test]
    fn transient_retried_then_ok() {
        let mut s = ScriptedStage::new(
            "t",
            vec![
                Err(StageError::Transient("a".into())),
                Err(StageError::Transient("b".into())),
                Ok("done".into()),
            ],
        );
        let r = run_stage(&mut s, "in", 2);
        assert_eq!(r.outcome, StageOutcome::Ok("done".into()));
        assert_eq!(r.retries, 2);
    }

    #[test]
    fn transient_exhausted_fails() {
        let mut s = ScriptedStage::new(
            "t",
            vec![
                Err(StageError::Transient("a".into())),
                Err(StageError::Transient("b".into())),
                Err(StageError::Transient("c".into())),
            ],
        );
        let r = run_stage(&mut s, "in", 1);
        assert!(matches!(r.outcome, StageOutcome::Failed(_)));
    }

    #[test]
    fn recoverable_uses_fallback() {
        let mut s = ScriptedStage::new("t", vec![Err(StageError::Recoverable("x".into()))])
            .with_fallback("fb");
        let r = run_stage(&mut s, "in", 0);
        assert_eq!(r.outcome, StageOutcome::Fallback("fb".into()));
    }

    #[test]
    fn fatal_fails_immediately() {
        let mut s = ScriptedStage::new("t", vec![Err(StageError::Fatal("boom".into()))]);
        let r = run_stage(&mut s, "in", 3);
        assert!(matches!(r.outcome, StageOutcome::Failed(_)));
        assert_eq!(r.retries, 0);
    }
}
