//! # apr tensors --stats — Aggregate Statistics Renderer
//!
//! `apr tensors --stats <FILE>` shows per-tensor (mean, std, min, max).
//! This recipe builds the renderer with deterministic column widths +
//! formatted display so CI logs are diff-friendly. Stats with NaN render
//! as `nan`, infinities as `+inf` / `-inf`.
//!
//! Demonstrates the **TENSORS.11** recipe for PMAT-110 (apr tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TENSORS-003
//!
//! Run with: cargo run --example cli_tensors_stats_aggregator
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct TensorRow {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

pub fn render_stat(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v == f64::INFINITY {
        "+inf".into()
    } else if v == f64::NEG_INFINITY {
        "-inf".into()
    } else {
        format!("{v:>+11.4e}")
    }
}

pub fn render_row(name: &str, row: TensorRow) -> String {
    format!(
        "{:<40} mean={} std={} min={} max={}",
        truncate_name(name, 40),
        render_stat(row.mean),
        render_stat(row.std),
        render_stat(row.min),
        render_stat(row.max),
    )
}

fn truncate_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        format!("{name:<width$}")
    } else {
        format!("…{}", &name[name.len() - (width - 1)..])
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tensors_stats_aggregator")?;

    let rows = [
        (
            "model.embed_tokens.weight",
            TensorRow {
                mean: 0.0,
                std: 0.02,
                min: -0.1,
                max: 0.1,
            },
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            TensorRow {
                mean: 0.001,
                std: 0.025,
                min: -0.15,
                max: 0.18,
            },
        ),
        (
            "nan_tensor",
            TensorRow {
                mean: f64::NAN,
                std: 0.0,
                min: 0.0,
                max: 0.0,
            },
        ),
        (
            "inf_tensor",
            TensorRow {
                mean: f64::INFINITY,
                std: 0.0,
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        ),
    ];
    for (name, row) in rows {
        println!("{}", render_row(name, row));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nan_renders_as_string() {
        assert_eq!(render_stat(f64::NAN), "nan");
    }

    #[test]
    fn pos_inf_renders_as_plus_inf() {
        assert_eq!(render_stat(f64::INFINITY), "+inf");
    }

    #[test]
    fn neg_inf_renders_as_minus_inf() {
        assert_eq!(render_stat(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn finite_renders_as_scientific_with_sign() {
        // 0.001 should render in scientific notation.
        let s = render_stat(0.001);
        assert!(s.contains('e'));
        // Positive value gets explicit + sign in our format.
        assert!(s.contains('+'));
    }

    #[test]
    fn name_truncation_preserves_suffix() {
        // Long names get truncated from the front, keeping the suffix —
        // because `model.layers.27.…` is more informative than `…27`.
        let truncated = truncate_name(&"a".repeat(100), 40);
        assert!(truncated.starts_with('…'));
        assert_eq!(truncated.chars().count(), 40);
    }

    #[test]
    fn short_name_padded_to_width() {
        let padded = truncate_name("short", 20);
        assert_eq!(padded.len(), 20);
        assert!(padded.starts_with("short"));
    }

    #[test]
    fn render_row_includes_all_four_stats() {
        let row = TensorRow {
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
        };
        let s = render_row("x", row);
        assert!(s.contains("mean="));
        assert!(s.contains("std="));
        assert!(s.contains("min="));
        assert!(s.contains("max="));
    }
}
