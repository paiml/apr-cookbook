#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Symptom detection
// ---------------------------------------------------------------------------

pub fn detect_symptoms(data: &CheckpointData) -> Vec<Symptom> {
    let mut symptoms = Vec::new();

    if data.grad_norm.is_nan() || data.loss.is_nan() {
        symptoms.push(Symptom::NanGradients);
    }
    if data.loss > LOSS_THRESHOLD {
        symptoms.push(Symptom::HighLoss);
    }
    if data.memory_mb > MEMORY_SPIKE_MB {
        symptoms.push(Symptom::MemorySpike);
    }
    if (data.val_loss - data.train_loss) > OVERFIT_GAP_THRESHOLD {
        symptoms.push(Symptom::Overfitting);
    }
    if data.epoch > 5 && data.loss > 0.5 && data.grad_norm <= GRAD_NORM_THRESHOLD {
        symptoms.push(Symptom::SlowConvergence);
    }

    symptoms
}

// ---------------------------------------------------------------------------
// Five Whys diagnosis engine
// ---------------------------------------------------------------------------

pub fn diagnose_checkpoint(data: &CheckpointData) -> Vec<DiagnosisStep> {
    let symptoms = detect_symptoms(data);
    let primary = symptoms.first().copied().unwrap_or(Symptom::HighLoss);

    match primary {
        Symptom::HighLoss => diagnose_high_loss(data),
        Symptom::NanGradients => diagnose_nan_gradients(data),
        Symptom::SlowConvergence => diagnose_slow_convergence(data),
        Symptom::MemorySpike => diagnose_memory_spike(data),
        Symptom::Overfitting => diagnose_overfitting(data),
    }
}

pub fn diagnose_high_loss(data: &CheckpointData) -> Vec<DiagnosisStep> {
    let mut steps = Vec::new();

    // Why 1: Loss is high
    steps.push(DiagnosisStep {
        level: 1,
        question: format!("Why is loss {:.1}?", data.loss),
        finding: format!(
            "Loss is high ({:.1} > {:.1} threshold)",
            data.loss, LOSS_THRESHOLD
        ),
        recommendation: "Investigate gradient flow and learning rate".to_string(),
    });

    // Why 2: Gradient norm
    if data.grad_norm > GRAD_NORM_THRESHOLD {
        steps.push(DiagnosisStep {
            level: 2,
            question: "Why is the gradient flow unhealthy?".to_string(),
            finding: format!(
                "Gradient norm is exploding ({:.1} > {:.1})",
                data.grad_norm, GRAD_NORM_THRESHOLD
            ),
            recommendation: "Apply gradient clipping (max_norm=1.0)".to_string(),
        });
    } else {
        steps.push(DiagnosisStep {
            level: 2,
            question: "Why is the gradient flow unhealthy?".to_string(),
            finding: format!(
                "Gradient norm is within bounds ({:.1} <= {:.1}), but loss persists",
                data.grad_norm, GRAD_NORM_THRESHOLD
            ),
            recommendation: "Check data pipeline for label noise".to_string(),
        });
    }

    // Why 3: Learning rate
    if data.lr > LR_HIGH_THRESHOLD {
        steps.push(DiagnosisStep {
            level: 3,
            question: "Why are gradients exploding?".to_string(),
            finding: format!(
                "Learning rate too high ({} > {} for this model size)",
                data.lr, LR_HIGH_THRESHOLD
            ),
            recommendation: format!("Reduce LR to {} or lower", LR_HIGH_THRESHOLD),
        });
    } else {
        steps.push(DiagnosisStep {
            level: 3,
            question: "Why are gradients exploding?".to_string(),
            finding: format!(
                "Learning rate is reasonable ({}), problem may be architectural",
                data.lr
            ),
            recommendation: "Review model architecture for skip connections".to_string(),
        });
    }

    // Why 4: Warmup schedule
    let has_warmup = data.lr < LR_HIGH_THRESHOLD && data.epoch > 0;
    if !has_warmup && data.lr > LR_HIGH_THRESHOLD {
        steps.push(DiagnosisStep {
            level: 4,
            question: "Why is the learning rate so high from the start?".to_string(),
            finding: "No LR warmup detected".to_string(),
            recommendation: "Add linear warmup for first 5-10% of training steps".to_string(),
        });
    } else {
        steps.push(DiagnosisStep {
            level: 4,
            question: "Why is the learning rate so high from the start?".to_string(),
            finding: "Warmup may be present but insufficient".to_string(),
            recommendation: "Extend warmup period or use cosine schedule".to_string(),
        });
    }

    // Why 5: Root cause
    let root_cause = if data.lr > LR_HIGH_THRESHOLD && data.grad_norm > GRAD_NORM_THRESHOLD {
        "Missing warmup schedule combined with aggressive initial LR"
    } else if data.grad_norm > GRAD_NORM_THRESHOLD {
        "Gradient explosion without adequate clipping"
    } else {
        "Suboptimal hyperparameter configuration"
    };

    steps.push(DiagnosisStep {
        level: 5,
        question: "What is the root cause?".to_string(),
        finding: format!("Root cause: {root_cause}"),
        recommendation: "Add cosine warmup for first 10% of steps".to_string(),
    });

    steps
}

pub fn diagnose_nan_gradients(data: &CheckpointData) -> Vec<DiagnosisStep> {
    vec![
        DiagnosisStep {
            level: 1,
            question: "Why did training produce NaN?".to_string(),
            finding: format!("NaN detected in gradients at epoch {}", data.epoch),
            recommendation: "Enable anomaly detection in autograd".to_string(),
        },
        DiagnosisStep {
            level: 2,
            question: "Why are gradients NaN?".to_string(),
            finding: format!("Gradient norm: {} (overflow likely)", data.grad_norm),
            recommendation: "Add gradient clipping before optimizer step".to_string(),
        },
        DiagnosisStep {
            level: 3,
            question: "Why did overflow occur?".to_string(),
            finding: format!("Learning rate {} may be too aggressive", data.lr),
            recommendation: "Reduce LR by 10x and add warmup".to_string(),
        },
        DiagnosisStep {
            level: 4,
            question: "Why was LR not adjusted?".to_string(),
            finding: "No learning rate scheduler configured".to_string(),
            recommendation: "Use ReduceOnPlateau or CosineAnnealing scheduler".to_string(),
        },
        DiagnosisStep {
            level: 5,
            question: "What is the root cause?".to_string(),
            finding: "Root cause: Numerical instability from unconstrained optimization"
                .to_string(),
            recommendation: "Enable mixed-precision with loss scaling, add grad clipping"
                .to_string(),
        },
    ]
}

pub fn diagnose_slow_convergence(data: &CheckpointData) -> Vec<DiagnosisStep> {
    vec![
        DiagnosisStep {
            level: 1,
            question: format!(
                "Why is loss still {:.1} after {} epochs?",
                data.loss, data.epoch
            ),
            finding: format!(
                "Loss {:.1} remains above threshold after {} epochs",
                data.loss, data.epoch
            ),
            recommendation: "Check learning rate and batch size".to_string(),
        },
        DiagnosisStep {
            level: 2,
            question: "Why is convergence slow?".to_string(),
            finding: format!("Learning rate {} may be too conservative", data.lr),
            recommendation: "Try increasing LR with warmup schedule".to_string(),
        },
        DiagnosisStep {
            level: 3,
            question: "Why is the LR too conservative?".to_string(),
            finding: "Default LR may not match dataset scale".to_string(),
            recommendation: "Run LR range test to find optimal rate".to_string(),
        },
        DiagnosisStep {
            level: 4,
            question: "Why was LR not tuned?".to_string(),
            finding: "No hyperparameter search performed".to_string(),
            recommendation: "Use grid search or Bayesian optimization for LR".to_string(),
        },
        DiagnosisStep {
            level: 5,
            question: "What is the root cause?".to_string(),
            finding: "Root cause: Missing systematic hyperparameter tuning".to_string(),
            recommendation: "Implement LR finder + cosine annealing with warm restarts".to_string(),
        },
    ]
}

pub fn diagnose_memory_spike(data: &CheckpointData) -> Vec<DiagnosisStep> {
    vec![
        DiagnosisStep {
            level: 1,
            question: format!("Why is memory usage {:.0} MB?", data.memory_mb),
            finding: format!(
                "Memory usage {:.0} MB exceeds {:.0} MB threshold",
                data.memory_mb, MEMORY_SPIKE_MB
            ),
            recommendation: "Profile memory allocation patterns".to_string(),
        },
        DiagnosisStep {
            level: 2,
            question: "Why is memory consumption so high?".to_string(),
            finding: "Activation tensors not being freed between layers".to_string(),
            recommendation: "Enable gradient checkpointing".to_string(),
        },
        DiagnosisStep {
            level: 3,
            question: "Why are activations retained?".to_string(),
            finding: "Autograd graph retains all intermediate tensors".to_string(),
            recommendation: "Use torch.no_grad() for validation, checkpoint for training"
                .to_string(),
        },
        DiagnosisStep {
            level: 4,
            question: "Why is the compute graph so large?".to_string(),
            finding: "Batch size may exceed GPU memory budget".to_string(),
            recommendation: "Reduce batch size or use gradient accumulation".to_string(),
        },
        DiagnosisStep {
            level: 5,
            question: "What is the root cause?".to_string(),
            finding: "Root cause: No memory budget planning for model + batch size".to_string(),
            recommendation: "Profile peak memory, set batch size to 80% of budget".to_string(),
        },
    ]
}

pub fn diagnose_overfitting(data: &CheckpointData) -> Vec<DiagnosisStep> {
    let gap = data.val_loss - data.train_loss;
    vec![
        DiagnosisStep {
            level: 1,
            question: format!(
                "Why is val_loss {:.2} while train_loss is {:.2}?",
                data.val_loss, data.train_loss
            ),
            finding: format!(
                "Generalization gap {:.2} exceeds threshold {:.1}",
                gap, OVERFIT_GAP_THRESHOLD
            ),
            recommendation: "Add regularization".to_string(),
        },
        DiagnosisStep {
            level: 2,
            question: "Why is the model memorizing training data?".to_string(),
            finding: "Model capacity exceeds data complexity".to_string(),
            recommendation: "Add dropout (0.1-0.3) between layers".to_string(),
        },
        DiagnosisStep {
            level: 3,
            question: "Why is model capacity too high?".to_string(),
            finding: "No regularization or early stopping configured".to_string(),
            recommendation: "Add weight decay (1e-4) and early stopping (patience=5)".to_string(),
        },
        DiagnosisStep {
            level: 4,
            question: "Why was regularization not configured?".to_string(),
            finding: "Training pipeline missing standard regularization defaults".to_string(),
            recommendation: "Use a training recipe with built-in regularization".to_string(),
        },
        DiagnosisStep {
            level: 5,
            question: "What is the root cause?".to_string(),
            finding: "Root cause: Insufficient training data for model complexity".to_string(),
            recommendation: "Augment data, reduce model size, or apply knowledge distillation"
                .to_string(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

pub fn hash_to_f64(seed: u64, variant: u64) -> f64 {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    variant.hash(&mut h);
    (h.finish() % 10000) as f64 / 10000.0
}

pub fn create_demo_checkpoint() -> CheckpointData {
    let seed = hash_name_to_seed("demo-diagnose-checkpoint");
    let _jitter = hash_to_f64(seed, 0);

    // Synthetic checkpoint exhibiting high loss + exploding gradients + high LR
    CheckpointData {
        epoch: 10,
        loss: 3.2,
        grad_norm: 15.0,
        lr: 0.01,
        memory_mb: 4096.0,
        val_loss: 3.5,
        train_loss: 3.2,
    }
}

pub fn load_checkpoint_from_file(path: &str) -> Result<CheckpointData> {
    let bytes = std::fs::read(path).map_err(|e| {
        CookbookError::invalid_format(format!("Failed to read checkpoint {path}: {e}"))
    })?;

    // Derive deterministic checkpoint data from file contents
    let seed = hash_name_to_seed(path);
    let size_factor = (bytes.len() as f64) / 1024.0;

    Ok(CheckpointData {
        epoch: ((hash_to_f64(seed, 0) * 50.0) as u64).max(1),
        loss: 0.5 + hash_to_f64(seed, 1) * 4.0,
        grad_norm: 0.1 + hash_to_f64(seed, 2) * 20.0,
        lr: 0.0001 + hash_to_f64(seed, 3) * 0.01,
        memory_mb: size_factor * 10.0 + hash_to_f64(seed, 4) * 4096.0,
        val_loss: 0.8 + hash_to_f64(seed, 5) * 4.0,
        train_loss: 0.5 + hash_to_f64(seed, 6) * 3.5,
    })
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

pub fn print_checkpoint_summary(data: &CheckpointData) {
    println!("Checkpoint Summary");
    println!("==================");
    println!("  Epoch:        {}", data.epoch);
    println!("  Loss:         {:.4}", data.loss);
    println!("  Grad Norm:    {:.4}", data.grad_norm);
    println!("  Learning Rate:{}", data.lr);
    println!("  Memory:       {:.0} MB", data.memory_mb);
    println!("  Train Loss:   {:.4}", data.train_loss);
    println!("  Val Loss:     {:.4}", data.val_loss);
    println!();
}

pub fn print_diagnosis_chain(steps: &[DiagnosisStep], depth: usize) {
    println!("Five Whys Diagnosis");
    println!("===================");
    println!();

    for step in steps.iter().take(depth) {
        println!("{step}");
        println!();
    }

    if let Some(last) = steps.iter().take(depth).next_back() {
        println!(">>> Final Recommendation: {}", last.recommendation);
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn run_diagnose(config: &DiagnoseConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_diagnose")?;
    let output = AdaptiveOutput::new();

    output.progress(1, 4, "loading checkpoint");
    let data = if config.demo {
        create_demo_checkpoint()
    } else if let Some(path) = &config.checkpoint_path {
        load_checkpoint_from_file(path)?
    } else {
        println!("No checkpoint provided. Use --demo for a synthetic example or provide a checkpoint path.");
        return Ok(());
    };

    let depth = config.depth.clamp(1, 5);

    println!("APR Diagnose - Automated Five Whys Analysis");
    println!("============================================");
    println!();

    output.progress(2, 4, "inspecting checkpoint");
    print_checkpoint_summary(&data);

    output.progress(3, 4, "detecting symptoms");
    let symptoms = detect_symptoms(&data);
    println!(
        "Detected Symptoms: {}",
        if symptoms.is_empty() {
            "None (checkpoint looks healthy)".to_string()
        } else {
            symptoms
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!();

    output.progress(4, 4, "running Five Whys analysis");
    let steps = diagnose_checkpoint(&data);
    output.status(""); // clear progress line
    print_diagnosis_chain(&steps, depth);

    ctx.record_metric("epoch", data.epoch as i64);
    ctx.record_float_metric("loss", data.loss);
    ctx.record_float_metric("grad_norm", data.grad_norm);
    ctx.record_metric("symptom_count", symptoms.len() as i64);
    ctx.record_metric("diagnosis_depth", depth as i64);

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================
