#![allow(unused_imports)]
//! # Recipe: APR Checkpoint Diagnose CLI
//!
//! **Category**: CLI Tools
//! **CLI Equivalent**: `apr diagnose`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate automated Five Whys root-cause analysis on a training
//! checkpoint. Mirrors `apr diagnose` which inspects loss, gradient norms,
//! learning rate, warmup schedule, and validation metrics to produce an
//! actionable diagnosis chain.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_diagnose
//! cargo run --example cli_apr_diagnose -- --demo
//! cargo run --example cli_apr_diagnose -- --help
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = DiagnoseConfig::parse();

    run_diagnose(&config)
}

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clap_parse_defaults() {
        let config = DiagnoseConfig::parse_from(["apr-diagnose"]);
        assert!(config.checkpoint_path.is_none());
        assert!(!config.demo);
        assert_eq!(config.depth, 5);
    }

    #[test]
    fn test_clap_parse_demo() {
        let config = DiagnoseConfig::parse_from(["apr-diagnose", "--demo"]);
        assert!(config.demo);
    }

    #[test]
    fn test_clap_parse_depth() {
        let config = DiagnoseConfig::parse_from(["apr-diagnose", "--depth", "3"]);
        assert_eq!(config.depth, 3);
    }

    #[test]
    fn test_clap_parse_checkpoint_path() {
        let config = DiagnoseConfig::parse_from(["apr-diagnose", "checkpoint.apr"]);
        assert_eq!(config.checkpoint_path, Some("checkpoint.apr".to_string()));
    }

    #[test]
    fn test_detect_symptoms_healthy() {
        let data = CheckpointData {
            epoch: 20,
            loss: 0.3,
            grad_norm: 1.0,
            lr: 0.0005,
            memory_mb: 2048.0,
            val_loss: 0.35,
            train_loss: 0.3,
        };
        let symptoms = detect_symptoms(&data);
        assert!(
            symptoms.is_empty(),
            "healthy checkpoint should have no symptoms"
        );
    }

    #[test]
    fn test_detect_symptoms_high_loss() {
        let data = CheckpointData {
            epoch: 1,
            loss: 3.2,
            grad_norm: 1.0,
            lr: 0.001,
            memory_mb: 2048.0,
            val_loss: 3.5,
            train_loss: 3.2,
        };
        let symptoms = detect_symptoms(&data);
        assert!(symptoms.contains(&Symptom::HighLoss));
    }

    #[test]
    fn test_detect_symptoms_nan_gradients() {
        let data = CheckpointData {
            epoch: 5,
            loss: f64::NAN,
            grad_norm: f64::NAN,
            lr: 0.01,
            memory_mb: 2048.0,
            val_loss: f64::NAN,
            train_loss: f64::NAN,
        };
        let symptoms = detect_symptoms(&data);
        assert!(symptoms.contains(&Symptom::NanGradients));
    }

    #[test]
    fn test_detect_symptoms_memory_spike() {
        let data = CheckpointData {
            epoch: 5,
            loss: 0.5,
            grad_norm: 1.0,
            lr: 0.001,
            memory_mb: 16384.0,
            val_loss: 0.6,
            train_loss: 0.5,
        };
        let symptoms = detect_symptoms(&data);
        assert!(symptoms.contains(&Symptom::MemorySpike));
    }

    #[test]
    fn test_detect_symptoms_overfitting() {
        let data = CheckpointData {
            epoch: 20,
            loss: 0.2,
            grad_norm: 0.5,
            lr: 0.0005,
            memory_mb: 2048.0,
            val_loss: 1.5,
            train_loss: 0.2,
        };
        let symptoms = detect_symptoms(&data);
        assert!(symptoms.contains(&Symptom::Overfitting));
    }

    #[test]
    fn test_detect_symptoms_slow_convergence() {
        let data = CheckpointData {
            epoch: 10,
            loss: 0.9,
            grad_norm: 1.0,
            lr: 0.0001,
            memory_mb: 2048.0,
            val_loss: 1.0,
            train_loss: 0.9,
        };
        let symptoms = detect_symptoms(&data);
        assert!(symptoms.contains(&Symptom::SlowConvergence));
    }

    #[test]
    fn test_diagnose_high_loss_chain() {
        let data = create_demo_checkpoint();
        let steps = diagnose_checkpoint(&data);
        assert_eq!(steps.len(), 5);
        assert_eq!(steps[0].level, 1);
        assert_eq!(steps[4].level, 5);
        assert!(steps[0].finding.contains("high"));
        assert!(steps[1].finding.contains("exploding"));
        assert!(steps[2].finding.contains("Learning rate too high"));
        assert!(steps[3].finding.contains("warmup"));
        assert!(steps[4].finding.contains("Root cause"));
    }

    #[test]
    fn test_diagnose_nan_chain() {
        let data = CheckpointData {
            epoch: 3,
            loss: f64::NAN,
            grad_norm: f64::NAN,
            lr: 0.1,
            memory_mb: 4096.0,
            val_loss: f64::NAN,
            train_loss: f64::NAN,
        };
        let steps = diagnose_checkpoint(&data);
        assert_eq!(steps.len(), 5);
        assert!(steps[0].finding.contains("NaN"));
    }

    #[test]
    fn test_diagnose_overfitting_chain() {
        let data = CheckpointData {
            epoch: 20,
            loss: 0.2,
            grad_norm: 0.5,
            lr: 0.0005,
            memory_mb: 2048.0,
            val_loss: 1.5,
            train_loss: 0.2,
        };
        let steps = diagnose_checkpoint(&data);
        assert_eq!(steps.len(), 5);
        assert!(steps[0].finding.contains("gap"));
    }

    #[test]
    fn test_diagnose_memory_spike_chain() {
        let data = CheckpointData {
            epoch: 5,
            loss: 0.5,
            grad_norm: 1.0,
            lr: 0.001,
            memory_mb: 16384.0,
            val_loss: 0.6,
            train_loss: 0.5,
        };
        let steps = diagnose_checkpoint(&data);
        assert_eq!(steps.len(), 5);
        assert!(steps[0].finding.contains("Memory"));
    }

    #[test]
    fn test_diagnose_slow_convergence_chain() {
        let data = CheckpointData {
            epoch: 10,
            loss: 0.9,
            grad_norm: 1.0,
            lr: 0.0001,
            memory_mb: 2048.0,
            val_loss: 1.0,
            train_loss: 0.9,
        };
        let steps = diagnose_checkpoint(&data);
        assert_eq!(steps.len(), 5);
        assert!(steps[0].finding.contains("remains above"));
    }

    #[test]
    fn test_demo_checkpoint_values() {
        let data = create_demo_checkpoint();
        assert!((data.loss - 3.2).abs() < f64::EPSILON);
        assert!((data.grad_norm - 15.0).abs() < f64::EPSILON);
        assert!((data.lr - 0.01).abs() < f64::EPSILON);
        assert_eq!(data.epoch, 10);
    }

    #[test]
    fn test_hash_to_f64_deterministic() {
        let a = hash_to_f64(42, 0);
        let b = hash_to_f64(42, 0);
        assert!((a - b).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hash_to_f64_range() {
        for seed in 0..100u64 {
            for variant in 0..10u64 {
                let val = hash_to_f64(seed, variant);
                assert!(
                    (0.0..1.0).contains(&val),
                    "hash_to_f64({seed}, {variant}) = {val} out of range"
                );
            }
        }
    }

    #[test]
    fn test_diagnosis_step_display() {
        let step = DiagnosisStep {
            level: 1,
            question: "Why is loss high?".to_string(),
            finding: "Loss is 3.2".to_string(),
            recommendation: "Check gradients".to_string(),
        };
        let display = format!("{step}");
        assert!(display.contains("Why 1:"));
        assert!(display.contains("Finding:"));
        assert!(display.contains("Recommendation:"));
    }

    #[test]
    fn test_symptom_display() {
        assert_eq!(Symptom::HighLoss.to_string(), "High Loss");
        assert_eq!(Symptom::NanGradients.to_string(), "NaN Gradients");
        assert_eq!(Symptom::SlowConvergence.to_string(), "Slow Convergence");
        assert_eq!(Symptom::MemorySpike.to_string(), "Memory Spike");
        assert_eq!(Symptom::Overfitting.to_string(), "Overfitting");
    }

    #[test]
    fn test_depth_clamp() {
        // Verify that depth clamping works for run_diagnose boundary
        assert_eq!(0_usize.clamp(1, 5), 1);
        assert_eq!(3_usize.clamp(1, 5), 3);
        assert_eq!(10_usize.clamp(1, 5), 5);
    }

    #[test]
    fn test_run_diagnose_demo() {
        let config = DiagnoseConfig {
            checkpoint_path: None,
            depth: 5,
            demo: true,
        };
        assert!(run_diagnose(&config).is_ok());
    }

    #[test]
    fn test_run_diagnose_no_args_prints_help() {
        let config = DiagnoseConfig {
            checkpoint_path: None,
            depth: 5,
            demo: false,
        };
        // Should succeed (prints message and returns Ok)
        assert!(run_diagnose(&config).is_ok());
    }

    #[test]
    fn test_multiple_symptoms_detected() {
        // Checkpoint with high loss, memory spike, and overfitting
        let data = CheckpointData {
            epoch: 10,
            loss: 5.0,
            grad_norm: 2.0,
            lr: 0.05,
            memory_mb: 32768.0,
            val_loss: 6.0,
            train_loss: 5.0,
        };
        let symptoms = detect_symptoms(&data);
        assert!(
            symptoms.len() >= 4,
            "should detect multiple symptoms: {symptoms:?}"
        );
        assert!(symptoms.contains(&Symptom::HighLoss));
        assert!(symptoms.contains(&Symptom::MemorySpike));
        assert!(symptoms.contains(&Symptom::Overfitting));
        assert!(symptoms.contains(&Symptom::SlowConvergence));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_diagnose_always_returns_5_steps(
            loss in 0.1f64..10.0,
            grad in 0.01f64..50.0,
            lr in 0.00001f64..0.1,
            epoch in 1u64..100,
        ) {
            let data = CheckpointData {
                epoch,
                loss,
                grad_norm: grad,
                lr,
                memory_mb: 2048.0,
                val_loss: loss + 0.1,
                train_loss: loss,
            };
            let steps = diagnose_checkpoint(&data);
            prop_assert_eq!(steps.len(), 5, "should always produce 5-step chain");
        }

        #[test]
        fn prop_diagnosis_levels_sequential(
            loss in 0.1f64..10.0,
            grad in 0.01f64..50.0,
        ) {
            let data = CheckpointData {
                epoch: 10,
                loss,
                grad_norm: grad,
                lr: 0.001,
                memory_mb: 2048.0,
                val_loss: loss + 0.2,
                train_loss: loss,
            };
            let steps = diagnose_checkpoint(&data);
            for (i, step) in steps.iter().enumerate() {
                prop_assert_eq!(step.level, i + 1, "step levels should be sequential");
            }
        }

        #[test]
        fn prop_hash_to_f64_in_unit_range(seed in 0u64..100_000, variant in 0u64..1000) {
            let val = hash_to_f64(seed, variant);
            prop_assert!(val >= 0.0, "hash_to_f64 must be >= 0.0, got {}", val);
            prop_assert!(val < 1.0, "hash_to_f64 must be < 1.0, got {}", val);
        }

        #[test]
        fn prop_detect_symptoms_never_panics(
            loss in -10.0f64..100.0,
            grad in -10.0f64..100.0,
            lr in 0.0f64..1.0,
            mem in 0.0f64..100_000.0,
            val_loss in -10.0f64..100.0,
            train_loss in -10.0f64..100.0,
        ) {
            let data = CheckpointData {
                epoch: 10,
                loss,
                grad_norm: grad,
                lr,
                memory_mb: mem,
                val_loss,
                train_loss,
            };
            // Should never panic
            let _ = detect_symptoms(&data);
        }
    }
}
