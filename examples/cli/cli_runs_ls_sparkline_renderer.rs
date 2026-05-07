//! # apr runs ls — Inline Sparkline Renderer
//!
//! `apr runs ls` shows a one-line summary per run with an inline sparkline
//! of loss values (▁▂▃▄▅▆▇█ characters). This recipe builds the renderer
//! and asserts the contract: deterministic per (values, width), values
//! mapped via min/max normalisation, empty input renders to a single-space.
//!
//! Demonstrates the **RUNS.4** recipe for PMAT-102 (apr runs ls coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender RUNS-001 + Tufte sparkline convention
//!
//! Run with: cargo run --example cli_runs_ls_sparkline_renderer
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SPARK_GLYPHS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub fn render_sparkline(values: &[f64], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return " ".to_string();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(f64::MIN_POSITIVE);

    let mut downsampled = Vec::with_capacity(width);
    let denom = width.saturating_sub(1).max(1);
    for i in 0..width {
        let idx = (i * values.len().saturating_sub(1)) / denom;
        downsampled.push(values[idx.min(values.len() - 1)]);
    }

    downsampled
        .iter()
        .map(|v| {
            let norm = (v - min) / range;
            let bucket = ((norm * (SPARK_GLYPHS.len() as f64 - 1.0)).round() as usize)
                .min(SPARK_GLYPHS.len() - 1);
            SPARK_GLYPHS[bucket]
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_runs_ls_sparkline_renderer")?;

    let losses_decay: Vec<f64> = (0..50).map(|i| 5.0 / (1.0 + i as f64 * 0.1)).collect();
    let losses_flat = vec![1.0; 50];
    let losses_spike = {
        let mut v = vec![1.0; 50];
        v[25] = 5.0;
        v
    };

    println!("decay:  [{}]", render_sparkline(&losses_decay, 24));
    println!("flat:   [{}]", render_sparkline(&losses_flat, 24));
    println!("spike:  [{}]", render_sparkline(&losses_spike, 24));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_yields_single_space() {
        assert_eq!(render_sparkline(&[], 10), " ");
    }

    #[test]
    fn zero_width_yields_single_space() {
        assert_eq!(render_sparkline(&[1.0, 2.0], 0), " ");
    }

    #[test]
    fn output_uses_only_spark_glyphs() {
        let s = render_sparkline(&[1.0, 2.0, 3.0], 5);
        for ch in s.chars() {
            assert!(SPARK_GLYPHS.contains(&ch), "non-spark glyph: {ch}");
        }
    }

    #[test]
    fn output_width_matches_requested() {
        let s = render_sparkline(&[1.0; 10], 7);
        assert_eq!(s.chars().count(), 7);
    }

    #[test]
    fn flat_input_renders_uniformly() {
        // All same values → all same glyph (any bucket).
        let s = render_sparkline(&[3.14; 10], 10);
        let first = s.chars().next().unwrap();
        assert!(s.chars().all(|c| c == first));
    }

    #[test]
    fn deterministic_for_same_input() {
        let v: Vec<f64> = (0..50).map(|i| (i as f64).sqrt()).collect();
        assert_eq!(render_sparkline(&v, 12), render_sparkline(&v, 12));
    }

    #[test]
    fn highest_value_maps_to_full_glyph() {
        // Last bucket = '█'. Max value should land there after normalisation.
        let s = render_sparkline(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6);
        assert!(s.chars().any(|c| c == '█'));
    }
}
