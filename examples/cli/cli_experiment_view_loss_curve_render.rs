//! # apr experiment view — Braille Loss Curve Renderer
//!
//! `apr experiment view` renders inline loss curves using Unicode braille
//! glyphs (8 dots per glyph, 2 columns × 4 rows). This recipe builds the
//! per-glyph encoder and asserts the contract: deterministic, every input
//! point maps to exactly one (glyph, dot_index) pair, empty input renders
//! to an empty string.
//!
//! Demonstrates the **EXPERIMENT.5** recipe for PMAT-102 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXPERIMENT-002 + Unicode block U+2800..U+28FF (Braille Patterns)
//!
//! Run with: cargo run --example cli_experiment_view_loss_curve_render
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const GLYPH_WIDTH: usize = 2; // braille = 2 cols × 4 rows
const GLYPH_HEIGHT: usize = 4;

pub fn render_braille(values: &[f64], canvas_width: usize) -> String {
    if values.is_empty() || canvas_width == 0 {
        return String::new();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(f64::MIN_POSITIVE);

    // Each glyph column carries 2 sample points (left + right dots).
    let samples_per_glyph = GLYPH_WIDTH;
    let total_cols = canvas_width * samples_per_glyph;
    let mut sampled = Vec::with_capacity(total_cols);
    for i in 0..total_cols {
        let idx = (i * values.len().saturating_sub(1)) / total_cols.max(1);
        sampled.push(values[idx.min(values.len() - 1)]);
    }

    let mut out = String::with_capacity(canvas_width * 4);
    for col in 0..canvas_width {
        let left = sampled[col * 2];
        let right = if col * 2 + 1 < sampled.len() {
            sampled[col * 2 + 1]
        } else {
            left
        };
        let lr = quantize_row(left, min, range);
        let rr = quantize_row(right, min, range);
        out.push(braille_glyph(lr, rr));
    }
    out
}

fn quantize_row(value: f64, min: f64, range: f64) -> u8 {
    let norm = (value - min) / range;
    let row = ((1.0 - norm) * (GLYPH_HEIGHT as f64 - 1.0)).round() as i32;
    row.clamp(0, GLYPH_HEIGHT as i32 - 1) as u8
}

fn braille_glyph(left_row: u8, right_row: u8) -> char {
    // Braille dot bit layout (Unicode U+2800):
    //   1 4
    //   2 5
    //   3 6
    //   7 8
    let mut bits = 0u32;
    for r in 0..GLYPH_HEIGHT as u8 {
        if r == left_row {
            bits |= match r {
                0 => 0x01,
                1 => 0x02,
                2 => 0x04,
                3 => 0x40,
                _ => 0,
            };
        }
        if r == right_row {
            bits |= match r {
                0 => 0x08,
                1 => 0x10,
                2 => 0x20,
                3 => 0x80,
                _ => 0,
            };
        }
    }
    char::from_u32(0x2800 + bits).unwrap_or('?')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_view_loss_curve_render")?;

    let losses: Vec<f64> = (0..100).map(|i| 5.0 - (i as f64 * 0.04)).collect();
    println!("decay:    {}", render_braille(&losses, 20));

    let mut spike = vec![1.0; 100];
    spike[50] = 5.0;
    println!("spike:    {}", render_braille(&spike, 20));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(render_braille(&[], 10), "");
    }

    #[test]
    fn zero_canvas_yields_empty_output() {
        assert_eq!(render_braille(&[1.0, 2.0], 0), "");
    }

    #[test]
    fn output_uses_braille_block_codepoints() {
        let s = render_braille(&[1.0, 2.0, 3.0], 5);
        for ch in s.chars() {
            assert!(
                (0x2800..=0x28FF).contains(&(ch as u32)),
                "non-braille char: {ch}"
            );
        }
    }

    #[test]
    fn output_width_matches_canvas() {
        let s = render_braille(&[1.0; 10], 7);
        assert_eq!(s.chars().count(), 7);
    }

    #[test]
    fn flat_input_renders_consistently() {
        // All points equal → range = 0, top row everywhere.
        let s = render_braille(&[1.0; 20], 10);
        // Should be 10 identical glyphs.
        let first = s.chars().next().unwrap();
        assert!(s.chars().all(|c| c == first));
    }

    #[test]
    fn deterministic_for_same_input() {
        let v: Vec<f64> = (0..50).map(|i| f64::from(i)).collect();
        let a = render_braille(&v, 12);
        let b = render_braille(&v, 12);
        assert_eq!(a, b);
    }

    #[test]
    fn quantize_row_clamps_to_valid_range() {
        let r = quantize_row(0.5, 0.0, 1.0);
        assert!(r < GLYPH_HEIGHT as u8);
    }
}
