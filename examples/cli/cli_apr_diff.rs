//! # Recipe: APR Model Diff CLI
//!
//! **Category**: CLI Tools
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
//! Compare two APR model files, showing differences in tensors, metadata,
//! and architecture. Detect weight drift between model versions.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_diff
//! cargo run --example cli_apr_diff -- --demo
//! ```

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args)?;

    if config.help {
        print_help();
        return Ok(());
    }

    run_diff(&config)
}

#[derive(Debug, Clone)]
struct DiffConfig {
    model_a: Option<String>,
    model_b: Option<String>,
    demo: bool,
    verbose: bool,
    threshold: f64,
    help: bool,
}

/// Snapshot of a model's structure and weights for comparison.
#[derive(Debug, Clone)]
struct ModelSnapshot {
    name: String,
    version: String,
    architecture: String,
    tensors: HashMap<String, TensorInfo>,
    total_size: usize,
}

/// Statistical information about a single tensor.
#[derive(Debug, Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    dtype: String,
    min: f64,
    max: f64,
    mean: f64,
    l2_norm: f64,
}

/// Status of a tensor in the diff.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TensorStatus {
    Added,
    Removed,
    Modified,
    Unchanged,
}

impl TensorStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Added => "ADDED",
            Self::Removed => "REMOVED",
            Self::Modified => "MODIFIED",
            Self::Unchanged => "UNCHANGED",
        }
    }

    fn symbol(&self) -> &'static str {
        match self {
            Self::Added => "+",
            Self::Removed => "-",
            Self::Modified => "~",
            Self::Unchanged => " ",
        }
    }
}

/// Diff result for a single tensor.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TensorDiff {
    name: String,
    status: TensorStatus,
    l2_distance: Option<f64>,
    shape_a: Option<Vec<usize>>,
    shape_b: Option<Vec<usize>>,
}

/// Complete diff report between two models.
#[derive(Debug, Clone)]
struct DiffReport {
    metadata_changes: Vec<String>,
    tensor_diffs: Vec<TensorDiff>,
    size_delta: i64,
    total_drift: f64,
}

fn parse_args(args: &[String]) -> Result<DiffConfig> {
    let mut config = DiffConfig {
        model_a: None,
        model_b: None,
        demo: false,
        verbose: false,
        threshold: 0.01,
        help: false,
    };

    let mut positional_count = 0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => config.help = true,
            "--demo" | "-d" => config.demo = true,
            "--verbose" | "-v" => config.verbose = true,
            "--threshold" | "-t" => {
                i += 1;
                if i < args.len() {
                    config.threshold = args[i].parse().unwrap_or(0.01);
                }
            }
            path if !path.starts_with('-') => {
                if positional_count == 0 {
                    config.model_a = Some(path.to_string());
                } else if positional_count == 1 {
                    config.model_b = Some(path.to_string());
                }
                positional_count += 1;
            }
            _ => {}
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("apr-diff - Compare two APR model files");
    println!();
    println!("USAGE:");
    println!("    apr-diff [OPTIONS] <MODEL_A> <MODEL_B>");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help             Print help information");
    println!("    -d, --demo             Run with demo models");
    println!("    -v, --verbose          Show detailed tensor info");
    println!("    -t, --threshold F      Drift threshold (default: 0.01)");
    println!();
    println!("EXAMPLES:");
    println!("    apr-diff model_v1.apr model_v2.apr");
    println!("    apr-diff --demo");
    println!("    apr-diff -v -t 0.001 base.apr finetuned.apr");
}

fn run_diff(config: &DiffConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_diff")?;

    // =========================================================================
    // Section 1: Create two model variants for comparison
    // =========================================================================

    let (snapshot_a, snapshot_b) = if config.demo {
        create_demo_snapshots()?
    } else if let (Some(a), Some(b)) = (&config.model_a, &config.model_b) {
        (load_snapshot(a)?, load_snapshot(b)?)
    } else {
        print_help();
        return Ok(());
    };

    println!("APR Model Diff");
    println!("==============");
    println!();
    println!(
        "Model A: {} (v{}, {})",
        snapshot_a.name, snapshot_a.version, snapshot_a.architecture
    );
    println!(
        "Model B: {} (v{}, {})",
        snapshot_b.name, snapshot_b.version, snapshot_b.architecture
    );
    println!();

    // =========================================================================
    // Section 2: Metadata diff (name, version, architecture)
    // =========================================================================

    let metadata_changes = diff_metadata(&snapshot_a, &snapshot_b);

    println!("Metadata Diff");
    println!("-------------");
    if metadata_changes.is_empty() {
        println!("  (no metadata changes)");
    } else {
        for change in &metadata_changes {
            println!("  {}", change);
        }
    }
    println!();

    // =========================================================================
    // Section 3: Tensor inventory diff (added/removed/common)
    // =========================================================================

    let (added, removed, common) = inventory_diff(&snapshot_a, &snapshot_b);

    println!("Tensor Inventory");
    println!("----------------");
    println!("  Model A tensors: {}", snapshot_a.tensors.len());
    println!("  Model B tensors: {}", snapshot_b.tensors.len());
    println!(
        "  Added:   {}  Removed: {}  Common: {}",
        added.len(),
        removed.len(),
        common.len()
    );
    println!();

    if !added.is_empty() {
        println!("  Added tensors:");
        for name in &added {
            if let Some(info) = snapshot_b.tensors.get(name) {
                println!("    + {} {:?} ({})", name, info.shape, info.dtype);
            }
        }
        println!();
    }

    if !removed.is_empty() {
        println!("  Removed tensors:");
        for name in &removed {
            if let Some(info) = snapshot_a.tensors.get(name) {
                println!("    - {} {:?} ({})", name, info.shape, info.dtype);
            }
        }
        println!();
    }

    // =========================================================================
    // Section 4: Weight drift analysis for common tensors
    // =========================================================================

    let tensor_diffs = analyze_drift(&snapshot_a, &snapshot_b, &common, config.threshold);

    println!("Weight Drift Analysis");
    println!("---------------------");

    let modified_count = tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Modified)
        .count();
    let unchanged_count = tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Unchanged)
        .count();

    println!(
        "  Modified: {}  Unchanged: {}  (threshold: {:.4})",
        modified_count, unchanged_count, config.threshold
    );
    println!();

    for diff in &tensor_diffs {
        if config.verbose || diff.status == TensorStatus::Modified {
            let distance_str = diff
                .l2_distance
                .map_or_else(|| "N/A".to_string(), |d| format!("{:.6}", d));
            println!(
                "  [{}] {} (L2 distance: {})",
                diff.status.symbol(),
                diff.name,
                distance_str
            );
        }
    }
    println!();

    // =========================================================================
    // Section 5: Size and compression comparison
    // =========================================================================

    let size_delta = snapshot_b.total_size as i64 - snapshot_a.total_size as i64;

    println!("Size Comparison");
    println!("---------------");
    println!(
        "  Model A: {} bytes ({:.2} KB)",
        snapshot_a.total_size,
        snapshot_a.total_size as f64 / 1024.0
    );
    println!(
        "  Model B: {} bytes ({:.2} KB)",
        snapshot_b.total_size,
        snapshot_b.total_size as f64 / 1024.0
    );

    let sign = if size_delta >= 0 { "+" } else { "" };
    println!("  Delta:   {}{} bytes", sign, size_delta);

    if snapshot_a.total_size > 0 {
        let pct = (size_delta as f64 / snapshot_a.total_size as f64) * 100.0;
        println!("  Change:  {}{:.2}%", sign, pct);
    }
    println!();

    // =========================================================================
    // Section 6: Diff summary report
    // =========================================================================

    let total_drift: f64 = tensor_diffs.iter().filter_map(|d| d.l2_distance).sum();

    let report = DiffReport {
        metadata_changes,
        tensor_diffs: build_full_diffs(&added, &removed, &tensor_diffs),
        size_delta,
        total_drift,
    };

    print_summary(&report, &snapshot_a, &snapshot_b);
    print_ascii_diff_visualization(&report);

    // Record metrics
    ctx.record_metric("metadata_changes", report.metadata_changes.len() as i64);
    ctx.record_metric("tensor_diffs", report.tensor_diffs.len() as i64);
    ctx.record_metric("size_delta", report.size_delta);
    ctx.record_float_metric("total_drift", report.total_drift);

    Ok(())
}

/// Create two demo model snapshots with known differences.
fn create_demo_snapshots() -> Result<(ModelSnapshot, ModelSnapshot)> {
    let seed_a = deterministic_seed("demo-model-v1");
    let seed_b = deterministic_seed("demo-model-v2");

    let mut tensors_a = HashMap::new();
    tensors_a.insert(
        "encoder.weight".to_string(),
        generate_tensor_info(&[768, 768], "fp32", seed_a, 0),
    );
    tensors_a.insert(
        "encoder.bias".to_string(),
        generate_tensor_info(&[768], "fp32", seed_a, 1),
    );
    tensors_a.insert(
        "decoder.weight".to_string(),
        generate_tensor_info(&[768, 768], "fp32", seed_a, 2),
    );
    tensors_a.insert(
        "decoder.bias".to_string(),
        generate_tensor_info(&[768], "fp32", seed_a, 3),
    );
    tensors_a.insert(
        "old_layer.weight".to_string(),
        generate_tensor_info(&[256, 256], "fp32", seed_a, 4),
    );

    let mut tensors_b = HashMap::new();
    tensors_b.insert(
        "encoder.weight".to_string(),
        generate_tensor_info(&[768, 768], "fp32", seed_b, 0),
    );
    tensors_b.insert(
        "encoder.bias".to_string(),
        generate_tensor_info(&[768], "fp32", seed_a, 1), // same as A (unchanged)
    );
    tensors_b.insert(
        "decoder.weight".to_string(),
        generate_tensor_info(&[768, 768], "fp32", seed_b, 2),
    );
    tensors_b.insert(
        "decoder.bias".to_string(),
        generate_tensor_info(&[768], "fp32", seed_b, 3),
    );
    tensors_b.insert(
        "new_head.weight".to_string(),
        generate_tensor_info(&[768, 10], "fp32", seed_b, 5),
    );

    let snapshot_a = ModelSnapshot {
        name: "demo-classifier".to_string(),
        version: "1.0.0".to_string(),
        architecture: "transformer".to_string(),
        total_size: compute_total_size(&tensors_a),
        tensors: tensors_a,
    };

    let snapshot_b = ModelSnapshot {
        name: "demo-classifier".to_string(),
        version: "2.0.0".to_string(),
        architecture: "transformer".to_string(),
        total_size: compute_total_size(&tensors_b),
        tensors: tensors_b,
    };

    Ok((snapshot_a, snapshot_b))
}

/// Load a model snapshot from a file path (simulated for demo).
fn load_snapshot(path: &str) -> Result<ModelSnapshot> {
    let bytes = std::fs::read(path)?;
    let seed = deterministic_seed(path);

    let mut tensors = HashMap::new();
    let n_tensors = (bytes.len() / 1024).clamp(1, 10);

    for i in 0..n_tensors {
        let name = format!("layer_{}.weight", i);
        let dim = 64 + (i * 32);
        tensors.insert(
            name,
            generate_tensor_info(&[dim, dim], "fp32", seed, i as u64),
        );
    }

    let file_stem = std::path::Path::new(path).file_stem().map_or_else(
        || "unknown".to_string(),
        |s| s.to_string_lossy().to_string(),
    );

    Ok(ModelSnapshot {
        name: file_stem,
        version: "1.0.0".to_string(),
        architecture: "linear".to_string(),
        total_size: bytes.len(),
        tensors,
    })
}

/// Generate deterministic tensor statistics from a seed.
fn generate_tensor_info(shape: &[usize], dtype: &str, seed: u64, index: u64) -> TensorInfo {
    let combined = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));

    let n_elements: usize = shape.iter().product();
    let scale = 1.0 / (n_elements as f64).sqrt();

    // Deterministic pseudo-random statistics using DefaultHasher
    let min = -scale * hash_to_float(combined, 0);
    let max = scale * hash_to_float(combined, 1);
    let mean = (min + max) / 2.0;
    let l2_norm = (n_elements as f64).sqrt() * scale * hash_to_float(combined, 2);

    TensorInfo {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        min,
        max,
        mean,
        l2_norm,
    }
}

/// Convert a seed into a deterministic float in [0.5, 1.5).
fn hash_to_float(seed: u64, variant: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    variant.hash(&mut hasher);
    let h = hasher.finish();
    0.5 + (h % 1000) as f64 / 1000.0
}

/// Deterministic seed from a name.
fn deterministic_seed(name: &str) -> u64 {
    hash_name_to_seed(name)
}

/// Compute total model size from tensor shapes.
fn compute_total_size(tensors: &HashMap<String, TensorInfo>) -> usize {
    tensors
        .values()
        .map(|t| {
            let n_elements: usize = t.shape.iter().product();
            let bytes_per_element = match t.dtype.as_str() {
                "fp32" => 4,
                "fp16" => 2,
                "int8" => 1,
                "int4" => 1, // approximation
                _ => 4,
            };
            n_elements * bytes_per_element
        })
        .sum()
}

/// Compare metadata fields between two snapshots.
fn diff_metadata(a: &ModelSnapshot, b: &ModelSnapshot) -> Vec<String> {
    let mut changes = Vec::new();

    if a.name != b.name {
        changes.push(format!("name: \"{}\" -> \"{}\"", a.name, b.name));
    }
    if a.version != b.version {
        changes.push(format!("version: \"{}\" -> \"{}\"", a.version, b.version));
    }
    if a.architecture != b.architecture {
        changes.push(format!(
            "architecture: \"{}\" -> \"{}\"",
            a.architecture, b.architecture
        ));
    }

    changes
}

/// Compute added, removed, and common tensor names.
fn inventory_diff(a: &ModelSnapshot, b: &ModelSnapshot) -> (Vec<String>, Vec<String>, Vec<String>) {
    let keys_a: std::collections::HashSet<&String> = a.tensors.keys().collect();
    let keys_b: std::collections::HashSet<&String> = b.tensors.keys().collect();

    let mut added: Vec<String> = keys_b.difference(&keys_a).map(|s| (*s).clone()).collect();
    let mut removed: Vec<String> = keys_a.difference(&keys_b).map(|s| (*s).clone()).collect();
    let mut common: Vec<String> = keys_a.intersection(&keys_b).map(|s| (*s).clone()).collect();

    added.sort();
    removed.sort();
    common.sort();

    (added, removed, common)
}

/// Analyze weight drift for common tensors.
fn analyze_drift(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    common: &[String],
    threshold: f64,
) -> Vec<TensorDiff> {
    common
        .iter()
        .map(|name| {
            let info_a = a.tensors.get(name);
            let info_b = b.tensors.get(name);

            match (info_a, info_b) {
                (Some(ta), Some(tb)) => {
                    let l2_distance = compute_l2_distance(ta, tb);
                    let status = if l2_distance > threshold {
                        TensorStatus::Modified
                    } else {
                        TensorStatus::Unchanged
                    };

                    TensorDiff {
                        name: name.clone(),
                        status,
                        l2_distance: Some(l2_distance),
                        shape_a: Some(ta.shape.clone()),
                        shape_b: Some(tb.shape.clone()),
                    }
                }
                _ => TensorDiff {
                    name: name.clone(),
                    status: TensorStatus::Unchanged,
                    l2_distance: None,
                    shape_a: None,
                    shape_b: None,
                },
            }
        })
        .collect()
}

/// Compute approximate L2 distance between two tensors from their statistics.
fn compute_l2_distance(a: &TensorInfo, b: &TensorInfo) -> f64 {
    let mean_diff = (a.mean - b.mean).powi(2);
    let norm_diff = (a.l2_norm - b.l2_norm).powi(2);
    let min_diff = (a.min - b.min).powi(2);
    let max_diff = (a.max - b.max).powi(2);

    (mean_diff + norm_diff + min_diff + max_diff).sqrt()
}

/// Build full diff list including added and removed tensors.
fn build_full_diffs(
    added: &[String],
    removed: &[String],
    common_diffs: &[TensorDiff],
) -> Vec<TensorDiff> {
    let mut all_diffs: Vec<TensorDiff> = Vec::new();

    for name in removed {
        all_diffs.push(TensorDiff {
            name: name.clone(),
            status: TensorStatus::Removed,
            l2_distance: None,
            shape_a: None,
            shape_b: None,
        });
    }

    all_diffs.extend(common_diffs.iter().cloned());

    for name in added {
        all_diffs.push(TensorDiff {
            name: name.clone(),
            status: TensorStatus::Added,
            l2_distance: None,
            shape_a: None,
            shape_b: None,
        });
    }

    all_diffs.sort_by(|a, b| a.name.cmp(&b.name));
    all_diffs
}

/// Print the summary report.
fn print_summary(report: &DiffReport, snapshot_a: &ModelSnapshot, snapshot_b: &ModelSnapshot) {
    println!("Diff Summary");
    println!("------------");

    let added_count = report
        .tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Added)
        .count();
    let removed_count = report
        .tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Removed)
        .count();
    let modified_count = report
        .tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Modified)
        .count();
    let unchanged_count = report
        .tensor_diffs
        .iter()
        .filter(|d| d.status == TensorStatus::Unchanged)
        .count();

    println!("  Metadata changes: {}", report.metadata_changes.len());
    println!(
        "  Tensors: {} added, {} removed, {} modified, {} unchanged",
        added_count, removed_count, modified_count, unchanged_count
    );

    let sign = if report.size_delta >= 0 { "+" } else { "" };
    println!("  Size delta: {}{} bytes", sign, report.size_delta);
    println!("  Total drift: {:.6}", report.total_drift);

    // Overall verdict
    let is_identical = report.metadata_changes.is_empty()
        && added_count == 0
        && removed_count == 0
        && modified_count == 0;

    println!();
    if is_identical {
        println!("  Verdict: IDENTICAL");
    } else if removed_count == 0 && modified_count <= snapshot_a.tensors.len() / 2 {
        println!("  Verdict: COMPATIBLE (minor changes)");
    } else {
        println!(
            "  Verdict: DIVERGED ({} structural changes)",
            added_count + removed_count
        );
    }
    println!();

    let _ = (snapshot_a, snapshot_b);
}

/// Print an ASCII visualization of the diff.
fn print_ascii_diff_visualization(report: &DiffReport) {
    println!("ASCII Diff");
    println!("----------");

    if report.tensor_diffs.is_empty() {
        println!("  (no tensors to compare)");
        println!();
        return;
    }

    // Find max name length for alignment
    let max_name_len = report
        .tensor_diffs
        .iter()
        .map(|d| d.name.len())
        .max()
        .unwrap_or(0);

    for diff in &report.tensor_diffs {
        let padded_name = format!("{:<width$}", diff.name, width = max_name_len);
        let bar = drift_bar(diff.l2_distance.unwrap_or(0.0));

        println!(
            "  {} {} |{}| {}",
            diff.status.symbol(),
            padded_name,
            bar,
            diff.status.as_str()
        );
    }
    println!();
}

/// Generate an ASCII bar representing drift magnitude.
fn drift_bar(distance: f64) -> String {
    let width = 20;
    // Clamp distance to [0, 1] range for visualization
    let clamped = distance.clamp(0.0, 1.0);
    let filled = (clamped * width as f64) as usize;
    let empty = width - filled;

    let mut bar = String::with_capacity(width);
    for _ in 0..filled {
        bar.push('#');
    }
    for _ in 0..empty {
        bar.push('.');
    }
    bar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_empty() {
        let args = vec!["apr-diff".to_string()];
        let config = parse_args(&args).expect("parse should succeed");
        assert!(config.model_a.is_none());
        assert!(config.model_b.is_none());
        assert!(!config.demo);
    }

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-diff".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse should succeed");
        assert!(config.demo);
    }

    #[test]
    fn test_parse_args_two_models() {
        let args = vec![
            "apr-diff".to_string(),
            "a.apr".to_string(),
            "b.apr".to_string(),
        ];
        let config = parse_args(&args).expect("parse should succeed");
        assert_eq!(config.model_a, Some("a.apr".to_string()));
        assert_eq!(config.model_b, Some("b.apr".to_string()));
    }

    #[test]
    fn test_parse_args_threshold() {
        let args = vec![
            "apr-diff".to_string(),
            "--threshold".to_string(),
            "0.05".to_string(),
        ];
        let config = parse_args(&args).expect("parse should succeed");
        assert!((config.threshold - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_parse_args_verbose() {
        let args = vec!["apr-diff".to_string(), "-v".to_string()];
        let config = parse_args(&args).expect("parse should succeed");
        assert!(config.verbose);
    }

    #[test]
    fn test_parse_args_help() {
        let args = vec!["apr-diff".to_string(), "-h".to_string()];
        let config = parse_args(&args).expect("parse should succeed");
        assert!(config.help);
    }

    #[test]
    fn test_create_demo_snapshots() {
        let (a, b) = create_demo_snapshots().expect("demo snapshots should succeed");
        assert_eq!(a.name, "demo-classifier");
        assert_eq!(b.name, "demo-classifier");
        assert_ne!(a.version, b.version);
        assert!(!a.tensors.is_empty());
        assert!(!b.tensors.is_empty());
    }

    #[test]
    fn test_diff_metadata_no_changes() {
        let a = ModelSnapshot {
            name: "model".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: HashMap::new(),
            total_size: 0,
        };
        let b = a.clone();
        let changes = diff_metadata(&a, &b);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_diff_metadata_all_changed() {
        let a = ModelSnapshot {
            name: "model-a".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: HashMap::new(),
            total_size: 0,
        };
        let b = ModelSnapshot {
            name: "model-b".to_string(),
            version: "2.0".to_string(),
            architecture: "transformer".to_string(),
            tensors: HashMap::new(),
            total_size: 0,
        };
        let changes = diff_metadata(&a, &b);
        assert_eq!(changes.len(), 3);
    }

    #[test]
    fn test_inventory_diff_added() {
        let mut tensors_a = HashMap::new();
        tensors_a.insert(
            "layer.weight".to_string(),
            generate_tensor_info(&[10, 10], "fp32", 42, 0),
        );

        let mut tensors_b = HashMap::new();
        tensors_b.insert(
            "layer.weight".to_string(),
            generate_tensor_info(&[10, 10], "fp32", 42, 0),
        );
        tensors_b.insert(
            "head.weight".to_string(),
            generate_tensor_info(&[10, 5], "fp32", 42, 1),
        );

        let a = ModelSnapshot {
            name: "a".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: tensors_a,
            total_size: 400,
        };
        let b = ModelSnapshot {
            name: "b".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: tensors_b,
            total_size: 600,
        };

        let (added, removed, common) = inventory_diff(&a, &b);
        assert_eq!(added.len(), 1);
        assert_eq!(added[0], "head.weight");
        assert!(removed.is_empty());
        assert_eq!(common.len(), 1);
    }

    #[test]
    fn test_inventory_diff_removed() {
        let mut tensors_a = HashMap::new();
        tensors_a.insert(
            "layer.weight".to_string(),
            generate_tensor_info(&[10, 10], "fp32", 42, 0),
        );
        tensors_a.insert(
            "old.weight".to_string(),
            generate_tensor_info(&[10], "fp32", 42, 1),
        );

        let mut tensors_b = HashMap::new();
        tensors_b.insert(
            "layer.weight".to_string(),
            generate_tensor_info(&[10, 10], "fp32", 42, 0),
        );

        let a = ModelSnapshot {
            name: "a".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: tensors_a,
            total_size: 440,
        };
        let b = ModelSnapshot {
            name: "b".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: tensors_b,
            total_size: 400,
        };

        let (added, removed, _common) = inventory_diff(&a, &b);
        assert!(added.is_empty());
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], "old.weight");
    }

    #[test]
    fn test_compute_l2_distance_identical() {
        let info = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        let distance = compute_l2_distance(&info, &info);
        assert!(distance.abs() < 1e-10);
    }

    #[test]
    fn test_compute_l2_distance_different() {
        let a = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        let b = generate_tensor_info(&[10, 10], "fp32", 99, 0);
        let distance = compute_l2_distance(&a, &b);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_analyze_drift_unchanged() {
        let info = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), info);

        let a = ModelSnapshot {
            name: "a".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors: tensors.clone(),
            total_size: 400,
        };
        let b = ModelSnapshot {
            name: "b".to_string(),
            version: "1.0".to_string(),
            architecture: "linear".to_string(),
            tensors,
            total_size: 400,
        };

        let common = vec!["w".to_string()];
        let diffs = analyze_drift(&a, &b, &common, 0.01);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].status, TensorStatus::Unchanged);
    }

    #[test]
    fn test_drift_bar_zero() {
        let bar = drift_bar(0.0);
        assert_eq!(bar.len(), 20);
        assert!(bar.chars().all(|c| c == '.'));
    }

    #[test]
    fn test_drift_bar_full() {
        let bar = drift_bar(1.0);
        assert_eq!(bar.len(), 20);
        assert!(bar.chars().all(|c| c == '#'));
    }

    #[test]
    fn test_drift_bar_half() {
        let bar = drift_bar(0.5);
        assert_eq!(bar.len(), 20);
        let filled = bar.chars().filter(|&c| c == '#').count();
        assert_eq!(filled, 10);
    }

    #[test]
    fn test_drift_bar_clamps_above_one() {
        let bar = drift_bar(5.0);
        assert_eq!(bar.len(), 20);
        assert!(bar.chars().all(|c| c == '#'));
    }

    #[test]
    fn test_compute_total_size() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorInfo {
                shape: vec![10, 10],
                dtype: "fp32".to_string(),
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                l2_norm: 1.0,
            },
        );
        // 10 * 10 * 4 bytes = 400
        assert_eq!(compute_total_size(&tensors), 400);
    }

    #[test]
    fn test_compute_total_size_fp16() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorInfo {
                shape: vec![10, 10],
                dtype: "fp16".to_string(),
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                l2_norm: 1.0,
            },
        );
        // 10 * 10 * 2 bytes = 200
        assert_eq!(compute_total_size(&tensors), 200);
    }

    #[test]
    fn test_build_full_diffs_sorted() {
        let added = vec!["z_tensor".to_string()];
        let removed = vec!["a_tensor".to_string()];
        let common = vec![TensorDiff {
            name: "m_tensor".to_string(),
            status: TensorStatus::Modified,
            l2_distance: Some(0.5),
            shape_a: Some(vec![10]),
            shape_b: Some(vec![10]),
        }];

        let full = build_full_diffs(&added, &removed, &common);
        assert_eq!(full.len(), 3);
        assert_eq!(full[0].name, "a_tensor");
        assert_eq!(full[1].name, "m_tensor");
        assert_eq!(full[2].name, "z_tensor");
    }

    #[test]
    fn test_tensor_status_symbols() {
        assert_eq!(TensorStatus::Added.symbol(), "+");
        assert_eq!(TensorStatus::Removed.symbol(), "-");
        assert_eq!(TensorStatus::Modified.symbol(), "~");
        assert_eq!(TensorStatus::Unchanged.symbol(), " ");
    }

    #[test]
    fn test_tensor_status_as_str() {
        assert_eq!(TensorStatus::Added.as_str(), "ADDED");
        assert_eq!(TensorStatus::Removed.as_str(), "REMOVED");
        assert_eq!(TensorStatus::Modified.as_str(), "MODIFIED");
        assert_eq!(TensorStatus::Unchanged.as_str(), "UNCHANGED");
    }

    #[test]
    fn test_hash_to_float_range() {
        for seed in 0..100 {
            for variant in 0..10 {
                let f = hash_to_float(seed, variant);
                assert!(
                    f >= 0.5,
                    "hash_to_float({}, {}) = {} < 0.5",
                    seed,
                    variant,
                    f
                );
                assert!(
                    f < 1.5,
                    "hash_to_float({}, {}) = {} >= 1.5",
                    seed,
                    variant,
                    f
                );
            }
        }
    }

    #[test]
    fn test_hash_to_float_deterministic() {
        let a = hash_to_float(42, 7);
        let b = hash_to_float(42, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_tensor_info_deterministic() {
        let a = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        let b = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.min, b.min);
        assert_eq!(a.max, b.max);
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.l2_norm, b.l2_norm);
    }

    #[test]
    fn test_demo_snapshots_have_differences() {
        let (a, b) = create_demo_snapshots().expect("demo should succeed");

        let (added, removed, common) = inventory_diff(&a, &b);
        assert!(!added.is_empty(), "demo should have added tensors");
        assert!(!removed.is_empty(), "demo should have removed tensors");
        assert!(!common.is_empty(), "demo should have common tensors");

        let diffs = analyze_drift(&a, &b, &common, 0.01);
        let modified = diffs.iter().any(|d| d.status == TensorStatus::Modified);
        assert!(modified, "demo should have modified tensors");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_l2_distance_non_negative(
            seed_a in 0u64..1000,
            seed_b in 0u64..1000,
            index in 0u64..10
        ) {
            let a = generate_tensor_info(&[10, 10], "fp32", seed_a, index);
            let b = generate_tensor_info(&[10, 10], "fp32", seed_b, index);
            let distance = compute_l2_distance(&a, &b);
            prop_assert!(distance >= 0.0);
        }

        #[test]
        fn prop_l2_distance_symmetric(
            seed_a in 0u64..1000,
            seed_b in 0u64..1000,
        ) {
            let a = generate_tensor_info(&[8, 8], "fp32", seed_a, 0);
            let b = generate_tensor_info(&[8, 8], "fp32", seed_b, 0);
            let d1 = compute_l2_distance(&a, &b);
            let d2 = compute_l2_distance(&b, &a);
            prop_assert!((d1 - d2).abs() < 1e-10);
        }

        #[test]
        fn prop_drift_bar_length(distance in -10.0f64..10.0) {
            let bar = drift_bar(distance);
            prop_assert_eq!(bar.len(), 20);
        }

        #[test]
        fn prop_inventory_diff_conservation(
            n_only_a in 0usize..5,
            n_only_b in 0usize..5,
            n_common in 0usize..5,
        ) {
            let mut tensors_a = HashMap::new();
            let mut tensors_b = HashMap::new();

            for i in 0..n_common {
                let name = format!("common_{}", i);
                let info = generate_tensor_info(&[4, 4], "fp32", 42, i as u64);
                tensors_a.insert(name.clone(), info.clone());
                tensors_b.insert(name, info);
            }
            for i in 0..n_only_a {
                let name = format!("only_a_{}", i);
                tensors_a.insert(name, generate_tensor_info(&[4, 4], "fp32", 42, (100 + i) as u64));
            }
            for i in 0..n_only_b {
                let name = format!("only_b_{}", i);
                tensors_b.insert(name, generate_tensor_info(&[4, 4], "fp32", 42, (200 + i) as u64));
            }

            let a = ModelSnapshot {
                name: "a".to_string(),
                version: "1.0".to_string(),
                architecture: "linear".to_string(),
                tensors: tensors_a,
                total_size: 0,
            };
            let b = ModelSnapshot {
                name: "b".to_string(),
                version: "1.0".to_string(),
                architecture: "linear".to_string(),
                tensors: tensors_b,
                total_size: 0,
            };

            let (added, removed, common) = inventory_diff(&a, &b);
            prop_assert_eq!(added.len(), n_only_b);
            prop_assert_eq!(removed.len(), n_only_a);
            prop_assert_eq!(common.len(), n_common);
        }
    }
}
