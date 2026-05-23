//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, clap::Parser)]
pub struct DiagnoseConfig {
    /// Checkpoint file path
    #[arg(value_name = "CHECKPOINT")]
    pub checkpoint_path: Option<String>,
    /// Maximum Why depth (range: 1-5)
    #[arg(long, default_value_t = 5)]
    pub depth: usize,
    /// Run with synthetic checkpoint (high loss scenario)
    #[arg(long)]
    pub demo: bool,
}

// ---------------------------------------------------------------------------
// Symptom classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symptom {
    HighLoss,
    NanGradients,
    SlowConvergence,
    MemorySpike,
    Overfitting,
}

impl fmt::Display for Symptom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighLoss => write!(f, "High Loss"),
            Self::NanGradients => write!(f, "NaN Gradients"),
            Self::SlowConvergence => write!(f, "Slow Convergence"),
            Self::MemorySpike => write!(f, "Memory Spike"),
            Self::Overfitting => write!(f, "Overfitting"),
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint and diagnosis types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CheckpointData {
    pub epoch: u64,
    pub loss: f64,
    pub grad_norm: f64,
    pub lr: f64,
    pub memory_mb: f64,
    pub val_loss: f64,
    pub train_loss: f64,
}

#[derive(Debug, Clone)]
pub struct DiagnosisStep {
    pub level: usize,
    pub question: String,
    pub finding: String,
    pub recommendation: String,
}

impl fmt::Display for DiagnosisStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indent = "  ".repeat(self.level);
        write!(
            f,
            "{indent}Why {}: {}\n{indent}  Finding: {}\n{indent}  Recommendation: {}",
            self.level, self.question, self.finding, self.recommendation
        )
    }
}

// ---------------------------------------------------------------------------
// Thresholds (rule-based)
// ---------------------------------------------------------------------------

pub const LOSS_THRESHOLD: f64 = 1.0;
pub const GRAD_NORM_THRESHOLD: f64 = 5.0;
pub const LR_HIGH_THRESHOLD: f64 = 0.001;
pub const MEMORY_SPIKE_MB: f64 = 8192.0;
pub const OVERFIT_GAP_THRESHOLD: f64 = 0.5;
// ---------------------------------------------------------------------------
// Symptom detection
// ---------------------------------------------------------------------------
