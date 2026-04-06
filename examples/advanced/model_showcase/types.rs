#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;
use std::time::Instant;

// ============================================================================
// Data Structures
// ============================================================================

/// Status of a single pipeline step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    // Step completed successfully.
    Done,
    // Step was skipped (precondition not met).
    Skip,
    // Step failed with an error.
    Fail,
}

impl fmt::Display for StepStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done => write!(f, "DONE"),
            Self::Skip => write!(f, "SKIP"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Result of a single showcase pipeline step.
#[derive(Debug, Clone)]
pub struct ShowcaseStep {
    // Human-readable step name.
    pub name: String,
    // Outcome of this step.
    pub status: StepStatus,
    // Wall-clock duration in milliseconds.
    pub duration_ms: f64,
    // Short detail string describing what happened.
    pub detail: String,
}

/// Aggregated report for the full showcase pipeline.
#[derive(Debug, Clone)]
pub struct ShowcaseReport {
    // Ordered list of pipeline steps.
    pub steps: Vec<ShowcaseStep>,
    // Name of the model under test.
    pub model_name: String,
}

impl ShowcaseReport {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            model_name: model_name.into(),
        }
    }

    pub fn push(&mut self, step: ShowcaseStep) {
        self.steps.push(step);
    }

    pub fn done_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Done)
            .count()
    }

    pub fn fail_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Fail)
            .count()
    }

    pub fn total_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.duration_ms).sum()
    }
}

/// Lightweight holder for bundle bytes produced in step 1.
pub struct CreatedModel {
    // Raw APR v2 bytes.
    pub bytes: Vec<u8>,
    // Number of parameters (weight elements).
    pub n_params: usize,
    // Number of tensors in the bundle.
    pub n_tensors: usize,
}

// ============================================================================
// Main Entry Point
// ============================================================================
