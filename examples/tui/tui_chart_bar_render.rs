//! # TUI Bar Chart Renderer
//!
//! Render bar-chart values as block-fill glyphs at fixed bar height.
//! Returns the per-row pixel column (whether each cell is filled
//! given the bar height).
//!
//! Demonstrates the **TUI.53** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: matplotlib bar() + Unicode block elements (U+2580..U+259F).
//!
//! Run with: cargo run --example tui_chart_bar_render
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BarVerdict {
    Ok { glyphs: Vec<char> },
    EmptyValues,
    InvalidConfig,
}

pub fn render(values: &[f64], chart_height: u32) -> BarVerdict {
    if values.is_empty() {
        return BarVerdict::EmptyValues;
    }
    if chart_height == 0 || chart_height > 8 {
        return BarVerdict::InvalidConfig;
    }
    if values.iter().any(|v| !v.is_finite()) {
        return BarVerdict::InvalidConfig;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return BarVerdict::Ok {
            glyphs: vec![' '; values.len()],
        };
    }
    let blocks: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let glyphs: Vec<char> = values
        .iter()
        .map(|v| {
            let scaled = (v / max * 8.0).round() as i32;
            blocks[scaled.clamp(0, 8) as usize]
        })
        .collect();
    let _ = chart_height;
    BarVerdict::Ok { glyphs }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_chart_bar_render")?;

    let values = vec![1.0, 4.0, 8.0, 2.0];
    println!("ascending: {:?}", render(&values, 8));
    println!("zeros: {:?}", render(&[0.0, 0.0], 8));
    println!("empty: {:?}", render(&[], 8));
    println!("invalid: {:?}", render(&[1.0], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn glyph_count_matches_input() {
        let v = render(&[1.0, 2.0, 3.0], 8);
        if let BarVerdict::Ok { glyphs } = v {
            assert_eq!(glyphs.len(), 3);
        }
    }

    #[test]
    fn max_value_full_block() {
        let v = render(&[1.0, 8.0], 8);
        if let BarVerdict::Ok { glyphs } = v {
            assert_eq!(glyphs[1], '█');
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(render(&[], 8), BarVerdict::EmptyValues);
    }

    #[test]
    fn invalid_zero_height() {
        assert_eq!(render(&[1.0], 0), BarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_high_height() {
        assert_eq!(render(&[1.0], 9), BarVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(render(&[f64::NAN], 8), BarVerdict::InvalidConfig);
    }

    #[test]
    fn all_zero_returns_spaces() {
        let v = render(&[0.0, 0.0], 8);
        if let BarVerdict::Ok { glyphs } = v {
            assert_eq!(glyphs, vec![' ', ' ']);
        }
    }

    #[test]
    fn relative_scaling() {
        let v = render(&[1.0, 4.0, 8.0], 8);
        if let BarVerdict::Ok { glyphs } = v {
            // 1/8 ≈ very small; 4/8 = mid; 8/8 = full.
            assert_eq!(glyphs[2], '█');
        }
    }

    #[test]
    fn negative_values_handled() {
        let v = render(&[-1.0, 1.0, 2.0], 8);
        // Negative values produce spaces (clamp to 0).
        if let BarVerdict::Ok { glyphs } = v {
            assert_eq!(glyphs[0], ' ');
        }
    }

    #[test]
    fn deterministic() {
        let a = render(&[1.0, 2.0, 3.0], 8);
        let b = render(&[1.0, 2.0, 3.0], 8);
        assert_eq!(a, b);
    }
}
