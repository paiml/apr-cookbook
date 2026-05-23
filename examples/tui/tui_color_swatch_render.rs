//! # TUI Color Swatch Render
//!
//! Render a horizontal color-swatch palette: each color appears as
//! a fixed-width block with hex label below. Returns rendered lines.
//!
//! Demonstrates the **TUI.175** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code color-picker swatches; Material Design palette
//!  10-color row layout.
//!
//! Run with: cargo run --example tui_color_swatch_render
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SwatchVerdict {
    Ok {
        swatch_line: String,
        label_line: String,
        block_width: u32,
    },
    InvalidConfig,
}

pub fn render(colors: &[(u8, u8, u8)], block_width: u32) -> SwatchVerdict {
    if colors.is_empty() || !(2..=20).contains(&block_width) {
        return SwatchVerdict::InvalidConfig;
    }
    let mut swatch = String::new();
    let mut labels = String::new();
    for (r, g, b) in colors {
        // Use solid block char × block_width for the swatch.
        let block: String = "█".repeat(block_width as usize);
        swatch.push_str(&block);
        let hex = format!("#{:02X}{:02X}{:02X}", r, g, b);
        // Pad label to block_width.
        let pad = block_width as usize;
        let label = if hex.len() <= pad {
            format!("{hex:<pad$}")
        } else {
            hex[..pad].to_string()
        };
        labels.push_str(&label);
    }
    SwatchVerdict::Ok {
        swatch_line: swatch,
        label_line: labels,
        block_width,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_color_swatch_render")?;

    let colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)];
    println!("rgb: {:?}", render(&colors, 8));
    println!("invalid: {:?}", render(&[], 5));
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
    fn empty_input_rejected() {
        assert_eq!(render(&[], 5), SwatchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_block_width_too_small() {
        assert_eq!(render(&[(0, 0, 0)], 1), SwatchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_block_width_too_large() {
        assert_eq!(render(&[(0, 0, 0)], 25), SwatchVerdict::InvalidConfig);
    }

    #[test]
    fn swatch_line_uses_block_chars() {
        let v = render(&[(255, 0, 0)], 5);
        if let SwatchVerdict::Ok { swatch_line, .. } = v {
            assert_eq!(swatch_line.matches('█').count(), 5);
        }
    }

    #[test]
    fn label_contains_hex() {
        let v = render(&[(255, 0, 0)], 8);
        if let SwatchVerdict::Ok { label_line, .. } = v {
            assert!(label_line.contains("#FF0000"));
        }
    }

    #[test]
    fn multiple_colors_concatenated() {
        let v = render(&[(255, 0, 0), (0, 255, 0)], 5);
        if let SwatchVerdict::Ok { swatch_line, .. } = v {
            assert_eq!(swatch_line.matches('█').count(), 10);
        }
    }

    #[test]
    fn block_width_returned() {
        let v = render(&[(0, 0, 0)], 7);
        if let SwatchVerdict::Ok { block_width, .. } = v {
            assert_eq!(block_width, 7);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&[(0, 0, 0)], 5);
        let r2 = render(&[(0, 0, 0)], 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_colors_handled() {
        let colors: Vec<(u8, u8, u8)> = (0..30).map(|_| (128, 128, 128)).collect();
        let v = render(&colors, 5);
        if let SwatchVerdict::Ok { swatch_line, .. } = v {
            assert_eq!(swatch_line.matches('█').count(), 150);
        }
    }

    #[test]
    fn high_color_values_handled() {
        let v = render(&[(255, 255, 255)], 8);
        if let SwatchVerdict::Ok { label_line, .. } = v {
            assert!(label_line.contains("FFFFFF"));
        }
    }

    #[test]
    fn black_color_rendered() {
        let v = render(&[(0, 0, 0)], 8);
        if let SwatchVerdict::Ok { label_line, .. } = v {
            assert!(label_line.contains("#000000"));
        }
    }

    #[test]
    fn min_block_width_accepted() {
        let v = render(&[(0, 0, 0)], 2);
        assert!(matches!(v, SwatchVerdict::Ok { .. }));
    }
}
