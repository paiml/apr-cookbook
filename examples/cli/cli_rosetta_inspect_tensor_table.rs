//! # apr rosetta inspect — Tensor Table Renderer
//!
//! `apr rosetta inspect <FILE>` lists the model's tensors in a fixed-width
//! table: `name (left-aligned, padded) | dtype | shape | bytes`. This
//! recipe builds the table renderer with deterministic column widths so
//! CI logs are diff-able. Width budget: name ≤ 60, dtype ≤ 8, shape ≤
//! 24, bytes ≤ 12.
//!
//! Demonstrates the **ROSETTA-INSPECT.3** recipe for PMAT-098 (apr rosetta inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001
//!
//! Run with: cargo run --example cli_rosetta_inspect_tensor_table
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone)]
pub struct TensorRow {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

const NAME_WIDTH: usize = 60;
const DTYPE_WIDTH: usize = 8;
const SHAPE_WIDTH: usize = 24;
const BYTES_WIDTH: usize = 12;

pub fn render_table(rows: &[TensorRow]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for r in rows {
        let name = if r.name.len() > NAME_WIDTH {
            format!("…{}", &r.name[r.name.len() - (NAME_WIDTH - 1)..])
        } else {
            r.name.clone()
        };
        let shape_str = format!("{:?}", r.shape);
        let _ = writeln!(
            out,
            "{:<name_w$} {:<dtype_w$} {:<shape_w$} {:>bytes_w$}",
            name,
            r.dtype,
            shape_str,
            r.bytes,
            name_w = NAME_WIDTH,
            dtype_w = DTYPE_WIDTH,
            shape_w = SHAPE_WIDTH,
            bytes_w = BYTES_WIDTH,
        );
    }
    out
}

pub fn total_bytes(rows: &[TensorRow]) -> u64 {
    rows.iter().map(|r| r.bytes).sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_inspect_tensor_table")?;

    let rows = vec![
        TensorRow {
            name: "model.embed_tokens.weight".into(),
            dtype: "bf16".into(),
            shape: vec![152064, 3584],
            bytes: 152_064 * 3584 * 2,
        },
        TensorRow {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: "bf16".into(),
            shape: vec![3584, 3584],
            bytes: 3584 * 3584 * 2,
        },
        TensorRow {
            name: "lm_head.weight".into(),
            dtype: "bf16".into(),
            shape: vec![152064, 3584],
            bytes: 152_064 * 3584 * 2,
        },
    ];

    println!("{}", render_table(&rows));
    println!("Total: {} bytes", total_bytes(&rows));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: &str, d: &str, s: Vec<u64>, b: u64) -> TensorRow {
        TensorRow {
            name: n.into(),
            dtype: d.into(),
            shape: s,
            bytes: b,
        }
    }

    #[test]
    fn table_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn each_row_yields_one_line() {
        let table = render_table(&[r("a", "fp16", vec![1, 2], 4)]);
        assert_eq!(table.lines().count(), 1);
    }

    #[test]
    fn long_name_truncated_with_ellipsis_prefix() {
        // Names > 60 chars get truncated to keep alignment.
        let long_name = "x".repeat(100);
        let table = render_table(&[r(&long_name, "fp16", vec![1], 2)]);
        let line = table.lines().next().unwrap();
        // Line must start with ellipsis sentinel.
        assert!(line.starts_with('…'));
    }

    #[test]
    fn short_name_left_padded_to_width() {
        let table = render_table(&[r("a", "fp16", vec![1], 2)]);
        let line = table.lines().next().unwrap();
        // Name "a" + 59 spaces = 60 chars, then space, then dtype starts.
        // First non-space after column 0: 'a'; gap before dtype is space-padding.
        assert!(line.starts_with('a'));
        // Dtype "fp16" appears after 60-char name + 1 separator space.
        let dtype_pos = line.find("fp16").unwrap();
        assert!(dtype_pos >= NAME_WIDTH);
    }

    #[test]
    fn total_bytes_sums() {
        let rows = vec![r("a", "fp16", vec![1], 2), r("b", "fp16", vec![1], 4)];
        assert_eq!(total_bytes(&rows), 6);
    }

    #[test]
    fn empty_rows_yield_empty_table() {
        assert!(render_table(&[]).is_empty());
        assert_eq!(total_bytes(&[]), 0);
    }

    #[test]
    fn shape_renders_with_brackets() {
        // Sanity: Debug on Vec<u64> renders as "[1, 2, 3]".
        let table = render_table(&[r("a", "fp16", vec![1, 2, 3], 24)]);
        assert!(table.contains("[1, 2, 3]"));
    }
}
