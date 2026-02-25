//! # APR Model Diff
//!
//! CLI equivalent: `apr diff model_a.apr model_b.apr --weights --values`
//!
//! Compares two APR models structurally and numerically. Reports tensor-level
//! weight differences including L2 distance, max absolute diff, mean absolute
//! diff, and cosine similarity. Essential for tracking fine-tuning impact.

use apr_cookbook::prelude::*;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TensorDiff {
    name: String,
    l2_distance: f64,
    max_abs_diff: f64,
    mean_abs_diff: f64,
    cosine_similarity: f64,
}

#[derive(Debug, Clone)]
struct StructuralChange {
    kind: ChangeKind,
    tensor_name: String,
    detail: String,
}

#[derive(Debug, Clone, PartialEq)]
enum ChangeKind {
    Added,
    Removed,
    ShapeChanged,
    Unchanged,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeKind::Added => write!(f, "ADDED"),
            ChangeKind::Removed => write!(f, "REMOVED"),
            ChangeKind::ShapeChanged => write!(f, "SHAPE_CHANGED"),
            ChangeKind::Unchanged => write!(f, "UNCHANGED"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DiffResult {
    structural_changes: Vec<StructuralChange>,
    weight_diffs: Vec<TensorDiff>,
}

// ---------------------------------------------------------------------------
// Diff logic
// ---------------------------------------------------------------------------

fn extract_float_payload(bytes: &[u8], offset: usize) -> Vec<f32> {
    let payload = &bytes[offset..];
    payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..n {
        let va = f64::from(a[i]);
        let vb = f64::from(b[i]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum: f64 = 0.0;
    for i in 0..n {
        let d = f64::from(a[i]) - f64::from(b[i]);
        sum += d * d;
    }
    sum.sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut max_d: f64 = 0.0;
    for i in 0..n {
        let d = (f64::from(a[i]) - f64::from(b[i])).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum += (f64::from(a[i]) - f64::from(b[i])).abs();
    }
    sum / n as f64
}

/// Simulated tensor directory: returns (name, start_offset, num_floats) triples.
/// In a real implementation, this parses the APR v2 tensor directory.
fn get_tensor_entries(bytes: &[u8]) -> Vec<(String, usize, usize)> {
    // For bundle-generated files, treat entire payload after 64-byte header
    // as a single weight tensor. Multiple tensors are concatenated sequentially.
    let header_size = 64.min(bytes.len());
    let payload_len = bytes.len() - header_size;
    let num_floats = payload_len / 4;

    if num_floats == 0 {
        return Vec::new();
    }

    // Split payload into named segments based on bundle size heuristic
    let segment_size = num_floats / 3; // assume ~3 tensors
    let remainder = num_floats - segment_size * 2;

    let mut entries = Vec::new();
    if segment_size > 0 {
        entries.push(("weight.0".to_string(), header_size, segment_size));
        entries.push((
            "weight.1".to_string(),
            header_size + segment_size * 4,
            segment_size,
        ));
        entries.push((
            "weight.2".to_string(),
            header_size + segment_size * 2 * 4,
            remainder,
        ));
    } else {
        entries.push(("weight".to_string(), header_size, num_floats));
    }
    entries
}

fn diff_models(a: &[u8], b: &[u8]) -> DiffResult {
    let entries_a = get_tensor_entries(a);
    let entries_b = get_tensor_entries(b);

    let mut structural_changes = Vec::new();
    let mut weight_diffs = Vec::new();

    // Build name sets
    let names_a: Vec<&str> = entries_a.iter().map(|(n, _, _)| n.as_str()).collect();
    let names_b: Vec<&str> = entries_b.iter().map(|(n, _, _)| n.as_str()).collect();

    // Check for removed tensors
    for name in &names_a {
        if !names_b.contains(name) {
            structural_changes.push(StructuralChange {
                kind: ChangeKind::Removed,
                tensor_name: (*name).to_string(),
                detail: "Tensor present in model A but not in model B".to_string(),
            });
        }
    }

    // Check for added tensors
    for name in &names_b {
        if !names_a.contains(name) {
            structural_changes.push(StructuralChange {
                kind: ChangeKind::Added,
                tensor_name: (*name).to_string(),
                detail: "Tensor present in model B but not in model A".to_string(),
            });
        }
    }

    // Compare common tensors
    for (name_a, offset_a, count_a) in &entries_a {
        if let Some((_, offset_b, count_b)) = entries_b.iter().find(|(n, _, _)| n == name_a) {
            if count_a == count_b {
                structural_changes.push(StructuralChange {
                    kind: ChangeKind::Unchanged,
                    tensor_name: name_a.clone(),
                    detail: format!("{count_a} params"),
                });
            } else {
                structural_changes.push(StructuralChange {
                    kind: ChangeKind::ShapeChanged,
                    tensor_name: name_a.clone(),
                    detail: format!("{count_a} params -> {count_b} params"),
                });
            }

            // Weight comparison
            let floats_a = extract_float_payload(a, *offset_a);
            let floats_b = extract_float_payload(b, *offset_b);
            let n = floats_a.len().min(floats_b.len());
            if n > 0 {
                let fa = &floats_a[..n];
                let fb = &floats_b[..n];
                weight_diffs.push(TensorDiff {
                    name: name_a.clone(),
                    l2_distance: l2_distance(fa, fb),
                    max_abs_diff: max_abs_diff(fa, fb),
                    mean_abs_diff: mean_abs_diff(fa, fb),
                    cosine_similarity: cosine_similarity(fa, fb),
                });
            }
        }
    }

    DiffResult {
        structural_changes,
        weight_diffs,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_diff")?;

    println!("=== APR Model Diff ===\n");

    // --- Section 1: Create base and fine-tuned models ---
    println!("--- Creating Base and Fine-Tuned Models ---");

    let dim = 64;
    let seed_base = hash_name_to_seed("diff-base");
    let seed_ft = hash_name_to_seed("diff-finetuned");

    let base_w1 = generate_model_payload(seed_base, dim * dim);
    let base_w2 = generate_model_payload(seed_base + 1, dim * 32);
    let base_w3 = generate_model_payload(seed_base + 2, 32);

    let bundle_a = ModelBundleV2::new()
        .with_name("base-model")
        .with_description("Base model before fine-tuning")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("encoder.weight", vec![dim, dim], base_w1)
        .add_tensor("decoder.weight", vec![dim, 32], base_w2)
        .add_tensor("decoder.bias", vec![32], base_w3)
        .build();

    // Fine-tuned model: same structure, slightly different weights
    let ft_w1 = generate_model_payload(seed_ft, dim * dim);
    let ft_w2 = generate_model_payload(seed_ft + 1, dim * 32);
    let ft_w3 = generate_model_payload(seed_ft + 2, 32);

    let bundle_b = ModelBundleV2::new()
        .with_name("finetuned-model")
        .with_description("Model after LoRA fine-tuning")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("encoder.weight", vec![dim, dim], ft_w1)
        .add_tensor("decoder.weight", vec![dim, 32], ft_w2)
        .add_tensor("decoder.bias", vec![32], ft_w3)
        .build();

    std::fs::write(ctx.path("base-model.apr"), &bundle_a)?;
    std::fs::write(ctx.path("finetuned-model.apr"), &bundle_b)?;
    println!("Base model:       {} bytes", bundle_a.len());
    println!("Fine-tuned model: {} bytes\n", bundle_b.len());

    // --- Section 2: Structural comparison ---
    println!("--- Structural Comparison ---");
    let diff = diff_models(&bundle_a, &bundle_b);

    println!("\n{:<20} {:<15} Detail", "Tensor", "Status");
    println!("{}", "-".repeat(60));
    for change in &diff.structural_changes {
        println!(
            "{:<20} {:<15} {}",
            change.tensor_name, change.kind, change.detail
        );
    }

    // --- Section 3: Per-tensor weight diff table ---
    println!("\n--- Weight Differences ---");
    println!(
        "\n{:<15} {:>12} {:>12} {:>12} {:>10}",
        "Tensor", "L2 Dist", "Max Abs", "Mean Abs", "Cosine"
    );
    println!("{}", "-".repeat(65));
    for wd in &diff.weight_diffs {
        println!(
            "{:<15} {:>12.6} {:>12.6} {:>12.6} {:>10.6}",
            wd.name, wd.l2_distance, wd.max_abs_diff, wd.mean_abs_diff, wd.cosine_similarity
        );
    }

    // --- Section 4: Summary statistics ---
    println!("\n--- Summary ---");
    if !diff.weight_diffs.is_empty() {
        let avg_l2: f64 = diff.weight_diffs.iter().map(|d| d.l2_distance).sum::<f64>()
            / diff.weight_diffs.len() as f64;
        let avg_cosine: f64 = diff
            .weight_diffs
            .iter()
            .map(|d| d.cosine_similarity)
            .sum::<f64>()
            / diff.weight_diffs.len() as f64;
        let max_max_abs: f64 = diff
            .weight_diffs
            .iter()
            .map(|d| d.max_abs_diff)
            .fold(0.0_f64, f64::max);

        println!("Average L2 distance:      {avg_l2:.6}");
        println!("Average cosine similarity: {avg_cosine:.6}");
        println!("Maximum absolute diff:     {max_max_abs:.6}");

        let structural_adds = diff
            .structural_changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Added)
            .count();
        let structural_removes = diff
            .structural_changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Removed)
            .count();
        println!("Tensors added:            {structural_adds}");
        println!("Tensors removed:          {structural_removes}");
    }

    // --- Section 5: Identical model diff ---
    println!("\n--- Self-Diff (identical models) ---");
    let self_diff = diff_models(&bundle_a, &bundle_a);
    for wd in &self_diff.weight_diffs {
        println!(
            "  {}: L2={:.6}, cosine={:.6}",
            wd.name, wd.l2_distance, wd.cosine_similarity
        );
        assert!(
            wd.l2_distance < 1e-10,
            "Identical models must have zero L2 distance"
        );
        assert!(
            (wd.cosine_similarity - 1.0).abs() < 1e-6,
            "Identical models must have cosine=1.0"
        );
    }
    println!("Self-diff verified: all distances are zero.");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bundle(name: &str, seed: u64, dim: usize) -> Vec<u8> {
        let payload = generate_model_payload(seed, dim * dim);
        ModelBundleV2::new()
            .with_name(name)
            .with_description("test")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![dim, dim], payload)
            .build()
    }

    #[test]
    fn test_identical_models_zero_diff() {
        let bundle = make_bundle("same", 42, 16);
        let diff = diff_models(&bundle, &bundle);
        for wd in &diff.weight_diffs {
            assert!(wd.l2_distance < 1e-10);
            assert!(wd.max_abs_diff < 1e-10);
            assert!(wd.mean_abs_diff < 1e-10);
        }
    }

    #[test]
    fn test_identical_models_cosine_one() {
        let bundle = make_bundle("cos", 42, 16);
        let diff = diff_models(&bundle, &bundle);
        for wd in &diff.weight_diffs {
            assert!((wd.cosine_similarity - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_different_models_nonzero_diff() {
        let a = make_bundle("a", 42, 16);
        let b = make_bundle("b", 99, 16);
        let diff = diff_models(&a, &b);
        let has_nonzero = diff.weight_diffs.iter().any(|d| d.l2_distance > 1e-6);
        assert!(has_nonzero, "Different models should have non-zero diff");
    }

    #[test]
    fn test_cosine_similarity_unit_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let cs = cosine_similarity(&a, &b);
        assert!((cs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let cs = cosine_similarity(&a, &b);
        assert!(cs.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_range() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![-1.0_f32, -2.0, -3.0, -4.0];
        let cs = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&cs));
    }

    #[test]
    fn test_l2_distance_zero_for_same() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let d = l2_distance(&a, &a);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_l2_distance_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![3.0_f32, 4.0, 0.0];
        let d = l2_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_abs_diff_known() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 5.0, 3.0];
        let d = max_abs_diff(&a, &b);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_abs_diff_known() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let d = mean_abs_diff(&a, &b);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_structural_unchanged() {
        let a = make_bundle("s", 42, 16);
        let b = make_bundle("s", 99, 16);
        let diff = diff_models(&a, &b);
        let has_unchanged = diff
            .structural_changes
            .iter()
            .any(|c| c.kind == ChangeKind::Unchanged);
        assert!(has_unchanged);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let cs = cosine_similarity(&[], &[]);
        assert!((cs - 0.0).abs() < 1e-6);
    }
}
