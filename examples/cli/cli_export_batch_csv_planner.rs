//! # apr export — `--batch` CSV Planner
//!
//! `apr export <FILE> --batch gguf,mlx,safetensors` plans multiple export
//! invocations from a comma-separated list. This recipe builds the parser
//! and per-format file-naming convention, asserting the contract:
//! dedup formats, reject unknowns, derive output filenames consistently.
//!
//! Demonstrates the **EXPORT.6** recipe for PMAT-099 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-EXPORT-002
//!
//! Run with: cargo run --example cli_export_batch_csv_planner
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPlan {
    pub formats: BTreeSet<String>,
    pub outputs: Vec<PathBuf>,
    pub unknown: Vec<String>,
}

const KNOWN_FORMATS: &[&str] = &["safetensors", "gguf", "mlx", "onnx", "openvino", "coreml"];

pub fn ext_for_format(format: &str) -> Option<&'static str> {
    match format {
        "safetensors" => Some("safetensors"),
        "gguf" => Some("gguf"),
        "mlx" => Some("mlx-bundle"),
        "onnx" => Some("onnx"),
        "openvino" => Some("openvino-ir"),
        "coreml" => Some("mlpackage"),
        _ => None,
    }
}

pub fn plan_batch(input_stem: &str, batch_csv: &str) -> ExportPlan {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut unknown = Vec::new();
    for raw in batch_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if KNOWN_FORMATS.contains(&raw) {
            seen.insert(raw.to_string());
        } else {
            unknown.push(raw.to_string());
        }
    }
    let outputs: Vec<PathBuf> = seen
        .iter()
        .filter_map(|f| ext_for_format(f).map(|ext| PathBuf::from(format!("{input_stem}.{ext}"))))
        .collect();
    ExportPlan {
        formats: seen,
        outputs,
        unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_batch_csv_planner")?;

    for csv in [
        "gguf,mlx,safetensors",
        "gguf, gguf , gguf",
        "torchscript,onnx",
        "",
    ] {
        let plan = plan_batch("model", csv);
        println!("--batch {csv:>30}");
        println!("  formats: {:?}", plan.formats);
        println!("  outputs: {:?}", plan.outputs);
        if !plan.unknown.is_empty() {
            println!("  unknown: {:?}", plan.unknown);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn three_distinct_formats_plan_three_outputs() {
        let plan = plan_batch("m", "gguf,mlx,safetensors");
        assert_eq!(plan.formats.len(), 3);
        assert_eq!(plan.outputs.len(), 3);
        assert!(plan.unknown.is_empty());
    }

    #[test]
    fn duplicates_deduped_in_plan() {
        // "gguf,gguf,gguf" → one entry only.
        let plan = plan_batch("m", "gguf,gguf,gguf");
        assert_eq!(plan.formats.len(), 1);
        assert_eq!(plan.outputs.len(), 1);
    }

    #[test]
    fn unknown_format_separated_from_known() {
        let plan = plan_batch("m", "torchscript,onnx,bogus");
        assert!(plan.formats.contains("onnx"));
        assert_eq!(plan.unknown.len(), 2);
    }

    #[test]
    fn empty_csv_yields_empty_plan() {
        let plan = plan_batch("m", "");
        assert!(plan.formats.is_empty());
        assert!(plan.outputs.is_empty());
        assert!(plan.unknown.is_empty());
    }

    #[test]
    fn whitespace_in_csv_trimmed() {
        let plan = plan_batch("m", " gguf ,   mlx  ,onnx");
        assert_eq!(plan.formats.len(), 3);
    }

    #[test]
    fn mlx_uses_bundle_directory_extension() {
        // MLX writes to a directory ending in .mlx-bundle, not a single file.
        let plan = plan_batch("m", "mlx");
        assert!(plan.outputs[0].to_string_lossy().ends_with(".mlx-bundle"));
    }

    #[test]
    fn coreml_uses_mlpackage_extension() {
        // Apple convention: .mlpackage for CoreML 5+.
        let plan = plan_batch("m", "coreml");
        assert!(plan.outputs[0].to_string_lossy().ends_with(".mlpackage"));
    }
}
