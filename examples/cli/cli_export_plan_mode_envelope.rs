//! # apr export — `--plan` Mode Envelope
//!
//! `apr export <FILE> --plan` validates inputs and shows the export plan
//! without writing any output. This recipe models the plan/run boundary:
//! when `--plan` is set the export must perform every validation but
//! must NOT touch the filesystem (no temp files, no output file). The
//! plan envelope shows: input dtype, target dtype, target file path,
//! estimated wall-clock, estimated output bytes.
//!
//! Demonstrates the **EXPORT.7** recipe for PMAT-099 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-EXPORT-003 + plan/apply convention
//!
//! Run with: cargo run --example cli_export_plan_mode_envelope
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct ExportPlan {
    pub input_path: String,
    pub output_path: String,
    pub source_dtype: String,
    pub target_dtype: String,
    pub estimated_seconds: u64,
    pub estimated_bytes: u64,
    pub side_effects: Vec<&'static str>, // empty in plan mode!
}

pub fn build_plan(
    input_path: &str,
    output_path: &str,
    source_dtype: &str,
    target_dtype: &str,
    input_bytes: u64,
    plan_mode: bool,
) -> ExportPlan {
    // Per-byte conversion cost, very rough.
    let throughput_bps = 100_000_000u64; // 100 MB/s
    let estimated_seconds = (input_bytes / throughput_bps).max(1);
    let estimated_bytes = match (source_dtype, target_dtype) {
        ("bf16" | "fp16", "fp32") => input_bytes * 2,
        ("fp32", "bf16" | "fp16") => input_bytes / 2,
        ("bf16" | "fp16", "int8") => input_bytes / 2,
        ("bf16" | "fp16", "int4") => input_bytes / 4,
        _ => input_bytes,
    };
    let side_effects = if plan_mode {
        Vec::new()
    } else {
        vec!["write output file", "write temp staging dir"]
    };
    ExportPlan {
        input_path: input_path.into(),
        output_path: output_path.into(),
        source_dtype: source_dtype.into(),
        target_dtype: target_dtype.into(),
        estimated_seconds,
        estimated_bytes,
        side_effects,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_plan_mode_envelope")?;

    let plan = build_plan(
        "model.apr",
        "model.gguf",
        "bf16",
        "int4",
        7_000_000_000, // 7 GB bf16 weights
        true,
    );
    println!("=== Plan mode ===");
    println!("{plan:#?}");

    let run = build_plan(
        "model.apr",
        "model.gguf",
        "bf16",
        "int4",
        7_000_000_000,
        false,
    );
    println!("\n=== Run mode (side effects) ===");
    println!("{:?}", run.side_effects);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn plan_mode_has_no_side_effects() {
        // Critical: --plan must NEVER write to disk.
        let p = build_plan("a.apr", "b.gguf", "bf16", "int4", 1_000_000, true);
        assert!(p.side_effects.is_empty());
    }

    #[test]
    fn run_mode_lists_side_effects() {
        // Run mode must surface what files will be touched.
        let p = build_plan("a.apr", "b.gguf", "bf16", "int4", 1_000_000, false);
        assert!(!p.side_effects.is_empty());
    }

    #[test]
    fn upcast_doubles_estimated_bytes() {
        let p = build_plan("a", "b", "bf16", "fp32", 1_000_000, true);
        assert_eq!(p.estimated_bytes, 2_000_000);
    }

    #[test]
    fn int4_downcast_quarters_estimated_bytes() {
        let p = build_plan("a", "b", "bf16", "int4", 1_000_000, true);
        assert_eq!(p.estimated_bytes, 250_000);
    }

    #[test]
    fn int8_downcast_halves_estimated_bytes() {
        let p = build_plan("a", "b", "fp16", "int8", 1_000_000, true);
        assert_eq!(p.estimated_bytes, 500_000);
    }

    #[test]
    fn estimated_seconds_at_least_one() {
        // Even tiny models report ≥1 second so the operator gets a usable ETA.
        let p = build_plan("a", "b", "bf16", "fp16", 100, true);
        assert!(p.estimated_seconds >= 1);
    }

    #[test]
    fn plan_preserves_paths_and_dtypes_verbatim() {
        let p = build_plan("input.apr", "output.gguf", "bf16", "int4", 1_000_000, true);
        assert_eq!(p.input_path, "input.apr");
        assert_eq!(p.output_path, "output.gguf");
        assert_eq!(p.source_dtype, "bf16");
        assert_eq!(p.target_dtype, "int4");
    }
}
