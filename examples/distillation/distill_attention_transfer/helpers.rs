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
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionModelSpec {
    pub name: String,
    pub layers: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub params_millions: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayerMapping {
    pub name: String,
    pub teacher_layer: u32,
    pub student_layer: u32,
    pub teacher_heads: u32,
    pub student_heads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionProjection {
    pub layer_name: String,
    pub teacher_attn_shape: (u32, u32), // (num_heads, seq_len)
    pub student_attn_shape: (u32, u32),
    pub projection_type: String,
    pub projection_params: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTransferConfig {
    pub epochs: u32,
    pub learning_rate: f64,
    pub beta: f64, // Weight for attention transfer loss
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochResult {
    pub epoch: u32,
    pub task_loss: f64,
    pub attention_transfer_loss: f64,
    pub total_loss: f64,
    pub teacher_accuracy: f64,
    pub student_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttentionQuality {
    pub layer_name: String,
    pub attention_mse: f64,
    pub cosine_similarity: f64,
    pub quality_label: String,
}

// --- Core Logic ---

pub fn build_layer_mappings(
    teacher: &AttentionModelSpec,
    student: &AttentionModelSpec,
) -> Result<Vec<AttentionLayerMapping>> {
    // Map student layers to evenly spaced teacher layers
    let student_layers = student.layers;
    let teacher_layers = teacher.layers;

    if student_layers == 0 {
        return Err(CookbookError::invalid_format(
            "Student must have at least one layer",
        ));
    }

    let names = ["early", "mid_early", "mid_late", "late"];

    let mappings: Vec<AttentionLayerMapping> = (0..student_layers)
        .map(|s_idx| {
            // Evenly distribute student layers across teacher layers
            let t_idx = if student_layers == 1 {
                teacher_layers - 1
            } else {
                (s_idx * (teacher_layers - 1)) / (student_layers - 1)
            };

            let name = names.get(s_idx as usize).copied().unwrap_or("extra");

            AttentionLayerMapping {
                name: name.to_string(),
                teacher_layer: t_idx,
                student_layer: s_idx,
                teacher_heads: teacher.num_heads,
                student_heads: student.num_heads,
            }
        })
        .collect();

    Ok(mappings)
}

pub fn compute_projection(mapping: &AttentionLayerMapping) -> Result<AttentionProjection> {
    let seq_len = 128_u32;

    let teacher_shape = (mapping.teacher_heads, seq_len);
    let student_shape = (mapping.student_heads, seq_len);

    // Projection type depends on whether head counts differ
    let (projection_type, projection_params) = if mapping.teacher_heads == mapping.student_heads {
        ("Identity".to_string(), 0)
    } else {
        // Linear projection: maps teacher heads -> student heads
        // Projection matrix shape: (student_heads, teacher_heads)
        let params = mapping.student_heads * mapping.teacher_heads;
        ("Linear".to_string(), params)
    };

    Ok(AttentionProjection {
        layer_name: mapping.name.clone(),
        teacher_attn_shape: teacher_shape,
        student_attn_shape: student_shape,
        projection_type,
        projection_params,
    })
}

pub fn simulate_attention_transfer_epoch(
    epoch: u32,
    config: &AttentionTransferConfig,
    projections: &[AttentionProjection],
) -> Result<EpochResult> {
    let progress = f64::from(epoch) / f64::from(config.epochs);

    // Teacher accuracy is constant (already trained)
    let teacher_accuracy = 0.92;

    // Task loss decreases over training
    let initial_task_loss = 2.0;
    let final_task_loss = 0.35;
    let task_loss = initial_task_loss - (initial_task_loss - final_task_loss) * progress;

    // Attention transfer loss: decreases as student learns teacher attention patterns
    // More projection layers = slightly harder to converge
    let n_projections = projections.len() as f64;
    let difficulty_factor = 1.0 + 0.05 * (n_projections - 1.0);
    let initial_attn_loss = 1.5 * difficulty_factor;
    let final_attn_loss = 0.08;
    let attention_transfer_loss = initial_attn_loss * (-2.5 * progress).exp()
        + final_attn_loss * (1.0 - (-2.5 * progress).exp());

    // Combined loss
    let total_loss = (1.0 - config.beta) * task_loss + config.beta * attention_transfer_loss;

    // Student accuracy: attention transfer provides a boost over vanilla distillation
    let max_student_accuracy = 0.895; // Better than vanilla (0.85) due to attention guidance
    let student_accuracy = max_student_accuracy * (1.0 - (-3.5 * progress).exp());

    Ok(EpochResult {
        epoch,
        task_loss,
        attention_transfer_loss,
        total_loss,
        teacher_accuracy,
        student_accuracy,
    })
}

pub fn compute_layer_attention_quality(
    proj: &AttentionProjection,
) -> Result<LayerAttentionQuality> {
    let seed = hash_name_to_seed(&proj.layer_name);

    // MSE decreases for later layers (better alignment after training)
    let base_mse = 0.15 - (seed % 10) as f64 / 100.0;
    let attention_mse = base_mse.max(0.02);

    // Cosine similarity of attention patterns
    let cosine_similarity = 0.85 + (seed % 12) as f64 / 100.0;
    let cosine_similarity = cosine_similarity.min(0.99);

    let quality_label = if cosine_similarity > 0.95 {
        "Excellent"
    } else if cosine_similarity > 0.90 {
        "Good"
    } else if cosine_similarity > 0.85 {
        "Fair"
    } else {
        "Needs work"
    };

    Ok(LayerAttentionQuality {
        layer_name: proj.layer_name.clone(),
        attention_mse,
        cosine_similarity,
        quality_label: quality_label.to_string(),
    })
}

pub fn save_log(path: &std::path::Path, log: &[EpochResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(log)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

// --- Tests ---
