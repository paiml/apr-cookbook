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
use clap::Parser;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Parser)]
#[command(name = "apr-diff", about = "Compare two APR model files")]
pub struct DiffConfig {
    // First model path
    pub model_a: Option<String>,
    // Second model path
    pub model_b: Option<String>,
    /// Demo mode
    #[arg(long, short = 'd')]
    pub demo: bool,
    /// Detailed info
    #[arg(long, short = 'v')]
    pub verbose: bool,
    /// Drift threshold
    #[arg(long, short = 't', default_value_t = 0.01)]
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub tensors: HashMap<String, TensorInfo>,
    pub total_size: usize,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub l2_norm: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorStatus {
    Added,
    Removed,
    Modified,
    Unchanged,
}
impl TensorStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Added => "ADDED",
            Self::Removed => "REMOVED",
            Self::Modified => "MODIFIED",
            Self::Unchanged => "UNCHANGED",
        }
    }
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Added => "+",
            Self::Removed => "-",
            Self::Modified => "~",
            Self::Unchanged => " ",
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorDiff {
    pub name: String,
    pub status: TensorStatus,
    pub l2_distance: Option<f64>,
    pub shape_a: Option<Vec<usize>>,
    pub shape_b: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct DiffReport {
    pub metadata_changes: Vec<String>,
    pub tensor_diffs: Vec<TensorDiff>,
    pub size_delta: i64,
    pub total_drift: f64,
}

pub fn count_by_status(diffs: &[TensorDiff], status: &TensorStatus) -> usize {
    diffs.iter().filter(|d| d.status == *status).count()
}

pub fn run_diff(config: &DiffConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_diff")?;
    let (sa, sb) = if config.demo {
        create_demo_snapshots()?
    } else if let (Some(a), Some(b)) = (&config.model_a, &config.model_b) {
        (load_snapshot(a)?, load_snapshot(b)?)
    } else {
        eprintln!("Error: provide two model paths or use --demo");
        return Ok(());
    };
    println!(
        "APR Model Diff\n==============\nModel A: {} (v{}, {})\nModel B: {} (v{}, {})\n",
        sa.name, sa.version, sa.architecture, sb.name, sb.version, sb.architecture
    );
    let mc = diff_metadata(&sa, &sb);
    println!(
        "Metadata: {}",
        if mc.is_empty() {
            "(no changes)".into()
        } else {
            mc.join("; ")
        }
    );
    let (added, removed, common) = inventory_diff(&sa, &sb);
    println!(
        "Tensors: A={}, B={}, added={}, removed={}, common={}",
        sa.tensors.len(),
        sb.tensors.len(),
        added.len(),
        removed.len(),
        common.len()
    );
    for n in &added {
        if let Some(i) = sb.tensors.get(n) {
            println!("  + {n} {:?} ({})", i.shape, i.dtype);
        }
    }
    for n in &removed {
        if let Some(i) = sa.tensors.get(n) {
            println!("  - {n} {:?} ({})", i.shape, i.dtype);
        }
    }
    let td = analyze_drift(&sa, &sb, &common, config.threshold);
    println!(
        "\nDrift: {} modified, {} unchanged (threshold: {:.4})",
        count_by_status(&td, &TensorStatus::Modified),
        count_by_status(&td, &TensorStatus::Unchanged),
        config.threshold
    );
    for d in &td {
        if config.verbose || d.status == TensorStatus::Modified {
            println!(
                "  [{}] {} (L2: {})",
                d.status.symbol(),
                d.name,
                d.l2_distance.map_or("N/A".into(), |v| format!("{v:.6}"))
            );
        }
    }
    let sd = sb.total_size as i64 - sa.total_size as i64;
    let sign = if sd >= 0 { "+" } else { "" };
    println!(
        "\nSize: A={} B={} delta={sign}{sd} bytes",
        sa.total_size, sb.total_size
    );
    let total_drift: f64 = td.iter().filter_map(|d| d.l2_distance).sum();
    let report = DiffReport {
        metadata_changes: mc,
        tensor_diffs: build_full_diffs(&added, &removed, &td),
        size_delta: sd,
        total_drift,
    };
    let ac = count_by_status(&report.tensor_diffs, &TensorStatus::Added);
    let rc = count_by_status(&report.tensor_diffs, &TensorStatus::Removed);
    let mc2 = count_by_status(&report.tensor_diffs, &TensorStatus::Modified);
    let uc = count_by_status(&report.tensor_diffs, &TensorStatus::Unchanged);
    println!(
        "\nSummary: {} meta, +{ac} -{rc} ~{mc2} ={uc} tensors, drift={:.6}",
        report.metadata_changes.len(),
        report.total_drift
    );
    let verdict = if report.metadata_changes.is_empty() && ac == 0 && rc == 0 && mc2 == 0 {
        "IDENTICAL"
    } else if rc == 0 && mc2 <= sa.tensors.len() / 2 {
        "COMPATIBLE"
    } else {
        "DIVERGED"
    };
    println!("Verdict: {verdict}");
    // ASCII viz
    let max_nl = report
        .tensor_diffs
        .iter()
        .map(|d| d.name.len())
        .max()
        .unwrap_or(0);
    for d in &report.tensor_diffs {
        let dist = d.l2_distance.unwrap_or(0.0).clamp(0.0, 1.0);
        let filled = (dist * 20.0) as usize;
        println!(
            "  {} {:<w$} |{}{}| {}",
            d.status.symbol(),
            d.name,
            "#".repeat(filled),
            ".".repeat(20 - filled),
            d.status.as_str(),
            w = max_nl
        );
    }
    ctx.record_metric("metadata_changes", report.metadata_changes.len() as i64);
    ctx.record_metric("tensor_diffs", report.tensor_diffs.len() as i64);
    ctx.record_metric("size_delta", report.size_delta);
    ctx.record_float_metric("total_drift", report.total_drift);
    Ok(())
}

pub fn create_demo_snapshots() -> Result<(ModelSnapshot, ModelSnapshot)> {
    let (sa, sb) = (
        deterministic_seed("demo-model-v1"),
        deterministic_seed("demo-model-v2"),
    );
    let ti = generate_tensor_info;
    let mut ta = HashMap::new();
    ta.insert("encoder.weight".into(), ti(&[768, 768], "fp32", sa, 0));
    ta.insert("encoder.bias".into(), ti(&[768], "fp32", sa, 1));
    ta.insert("decoder.weight".into(), ti(&[768, 768], "fp32", sa, 2));
    ta.insert("decoder.bias".into(), ti(&[768], "fp32", sa, 3));
    ta.insert("old_layer.weight".into(), ti(&[256, 256], "fp32", sa, 4));
    let mut tb = HashMap::new();
    tb.insert("encoder.weight".into(), ti(&[768, 768], "fp32", sb, 0));
    tb.insert("encoder.bias".into(), ti(&[768], "fp32", sa, 1));
    tb.insert("decoder.weight".into(), ti(&[768, 768], "fp32", sb, 2));
    tb.insert("decoder.bias".into(), ti(&[768], "fp32", sb, 3));
    tb.insert("new_head.weight".into(), ti(&[768, 10], "fp32", sb, 5));
    Ok((
        ModelSnapshot {
            name: "demo-classifier".into(),
            version: "1.0.0".into(),
            architecture: "transformer".into(),
            total_size: compute_total_size(&ta),
            tensors: ta,
        },
        ModelSnapshot {
            name: "demo-classifier".into(),
            version: "2.0.0".into(),
            architecture: "transformer".into(),
            total_size: compute_total_size(&tb),
            tensors: tb,
        },
    ))
}

pub fn load_snapshot(path: &str) -> Result<ModelSnapshot> {
    let bytes = std::fs::read(path)?;
    let seed = deterministic_seed(path);
    let mut tensors = HashMap::new();
    for i in 0..(bytes.len() / 1024).clamp(1, 10) {
        tensors.insert(
            format!("layer_{i}.weight"),
            generate_tensor_info(&[64 + i * 32, 64 + i * 32], "fp32", seed, i as u64),
        );
    }
    let name = std::path::Path::new(path)
        .file_stem()
        .map_or("unknown".into(), |s| s.to_string_lossy().into());
    Ok(ModelSnapshot {
        name,
        version: "1.0.0".into(),
        architecture: "linear".into(),
        total_size: bytes.len(),
        tensors,
    })
}

pub fn generate_tensor_info(shape: &[usize], dtype: &str, seed: u64, index: u64) -> TensorInfo {
    let c = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let n: usize = shape.iter().product();
    let s = 1.0 / (n as f64).sqrt();
    TensorInfo {
        shape: shape.to_vec(),
        dtype: dtype.into(),
        min: -s * hash_to_float(c, 0),
        max: s * hash_to_float(c, 1),
        mean: 0.0,
        l2_norm: (n as f64).sqrt() * s * hash_to_float(c, 2),
    }
}

pub fn hash_to_float(seed: u64, variant: u64) -> f64 {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    variant.hash(&mut h);
    0.5 + (h.finish() % 1000) as f64 / 1000.0
}
pub fn deterministic_seed(name: &str) -> u64 {
    hash_name_to_seed(name)
}
pub fn compute_total_size(t: &HashMap<String, TensorInfo>) -> usize {
    t.values()
        .map(|i| {
            let n: usize = i.shape.iter().product();
            n * match i.dtype.as_str() {
                "fp16" => 2,
                "int8" | "int4" => 1,
                _ => 4,
            }
        })
        .sum()
}

pub fn diff_metadata(a: &ModelSnapshot, b: &ModelSnapshot) -> Vec<String> {
    let mut c = Vec::new();
    if a.name != b.name {
        c.push(format!("name: \"{}\" -> \"{}\"", a.name, b.name));
    }
    if a.version != b.version {
        c.push(format!("version: \"{}\" -> \"{}\"", a.version, b.version));
    }
    if a.architecture != b.architecture {
        c.push(format!(
            "arch: \"{}\" -> \"{}\"",
            a.architecture, b.architecture
        ));
    }
    c
}

pub fn inventory_diff(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (ka, kb): (
        std::collections::HashSet<&String>,
        std::collections::HashSet<&String>,
    ) = (a.tensors.keys().collect(), b.tensors.keys().collect());
    let mut added: Vec<String> = kb.difference(&ka).map(|s| (*s).clone()).collect();
    added.sort();
    let mut removed: Vec<String> = ka.difference(&kb).map(|s| (*s).clone()).collect();
    removed.sort();
    let mut common: Vec<String> = ka.intersection(&kb).map(|s| (*s).clone()).collect();
    common.sort();
    (added, removed, common)
}

pub fn analyze_drift(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    common: &[String],
    threshold: f64,
) -> Vec<TensorDiff> {
    common
        .iter()
        .map(|name| match (a.tensors.get(name), b.tensors.get(name)) {
            (Some(ta), Some(tb)) => {
                let d = ((ta.mean - tb.mean).powi(2)
                    + (ta.l2_norm - tb.l2_norm).powi(2)
                    + (ta.min - tb.min).powi(2)
                    + (ta.max - tb.max).powi(2))
                .sqrt();
                TensorDiff {
                    name: name.clone(),
                    status: if d > threshold {
                        TensorStatus::Modified
                    } else {
                        TensorStatus::Unchanged
                    },
                    l2_distance: Some(d),
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
        })
        .collect()
}

pub fn build_full_diffs(
    added: &[String],
    removed: &[String],
    common: &[TensorDiff],
) -> Vec<TensorDiff> {
    let mut all: Vec<TensorDiff> = removed
        .iter()
        .map(|n| TensorDiff {
            name: n.clone(),
            status: TensorStatus::Removed,
            l2_distance: None,
            shape_a: None,
            shape_b: None,
        })
        .collect();
    all.extend(common.iter().cloned());
    all.extend(added.iter().map(|n| TensorDiff {
        name: n.clone(),
        status: TensorStatus::Added,
        l2_distance: None,
        shape_a: None,
        shape_b: None,
    }));
    all.sort_by(|a, b| a.name.cmp(&b.name));
    all
}
