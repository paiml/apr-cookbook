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
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

impl OnlineTrainingPipeline {
    pub fn train(&mut self, trace: &ExecutionTrace) {
        let Some(label) = trace.has_defect else {
            return;
        };
        let features = trace.to_features();
        let predicted = if self.use_sgd {
            self.sgd.predict(&features)
        } else {
            self.pa.predict(&features)
        };
        self.metrics.update(predicted, label);
        self.drift.add_error(predicted, label);
        match self.drift.detect_drift() {
            DriftStatus::Drift => {
                self.use_sgd = !self.use_sgd;
                self.drift.reset();
            }
            DriftStatus::Warning => {
                self.sgd.update(&features, label);
                self.pa.update(&features, label);
                return;
            }
            DriftStatus::Stable => {}
        }
        if self.use_sgd {
            self.sgd.update(&features, label);
        } else {
            self.pa.update(&features, label);
        }
    }

    #[must_use]
    pub fn predict(&self, trace: &ExecutionTrace) -> DefectPrediction {
        let features = trace.to_features();
        let (probability, confidence) = if self.use_sgd {
            let p = self.sgd.predict_proba(&features);
            (p, (p - 0.5).abs() * 2.0)
        } else {
            let s = self.pa.predict_score(&features);
            (sigmoid(s), s.abs().min(1.0))
        };
        DefectPrediction {
            is_defect: probability > 0.5,
            probability,
            confidence,
            model_type: if self.use_sgd { "SGD" } else { "PA" },
        }
    }
}

impl PipelineMetrics {
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f32 / t as f32
        }
    }
    #[must_use]
    pub fn precision(&self) -> f32 {
        let d = self.true_positives + self.false_positives;
        if d == 0 {
            0.0
        } else {
            self.true_positives as f32 / d as f32
        }
    }
    #[must_use]
    pub fn recall(&self) -> f32 {
        let d = self.true_positives + self.false_negatives;
        if d == 0 {
            0.0
        } else {
            self.true_positives as f32 / d as f32
        }
    }
    #[must_use]
    pub fn f1_score(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
    #[must_use]
    pub fn total(&self) -> u64 {
        self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    }
}

impl TraceGenerator {
    pub fn generate(&mut self) -> ExecutionTrace {
        let is_defect = self.rng.next_f32() < self.defect_rate;
        let mut t = ExecutionTrace::new();
        if is_defect {
            match self.rng.next_u64() % 4 {
                0 => {
                    t.memory_allocated = 1_000_000 + self.rng.next_u64() % 10_000_000;
                    t.memory_freed = t.memory_allocated / 10;
                }
                1 => {
                    t.call_count = 500_000 + self.rng.next_u64() % 1_000_000;
                    t.io_ops = 0;
                    t.execution_time_us = 5_000_000;
                }
                2 => {
                    t.max_depth = 500 + (self.rng.next_u64() % 500) as u32;
                    t.call_count = 10000;
                }
                _ => {
                    t.io_ops = 50000 + (self.rng.next_u64() % 50000) as u32;
                    t.execution_time_us = 10_000_000;
                }
            }
        } else {
            t.call_count = 100 + self.rng.next_u64() % 10000;
            t.max_depth = 5 + (self.rng.next_u64() % 20) as u32;
            t.memory_allocated = 10000 + self.rng.next_u64() % 100000;
            t.memory_freed = t.memory_allocated - self.rng.next_u64() % 1000;
            t.execution_time_us = 1000 + self.rng.next_u64() % 50000;
            t.io_ops = (self.rng.next_u64() % 100) as u32;
            t.branch_misses = (self.rng.next_u64() % 1000) as u32;
        }
        t.has_defect = Some(is_defect);
        t
    }
}
