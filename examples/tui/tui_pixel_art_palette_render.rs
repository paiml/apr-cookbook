//! # TUI Pixel Art Palette Render
//!
//! Render a small pixel-art glyph from a 2D color matrix using palette
//! characters. Returns rendered lines.
//!
//! Demonstrates the **TUI.117** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ASCII art conventions (Hewlett-Packard PCL5 line-printer
//!  graphics); 8-bit color palettes (NES/Atari).
//!
//! Run with: cargo run --example tui_pixel_art_palette_render
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PixelArtVerdict {
    Ok {
        lines: Vec<String>,
        width: u32,
        height: u32,
    },
    InvalidConfig,
}

pub fn render(pixels: &[Vec<u8>], palette: &[char]) -> PixelArtVerdict {
    if pixels.is_empty() || palette.is_empty() {
        return PixelArtVerdict::InvalidConfig;
    }
    let w = pixels[0].len();
    if w == 0 || pixels.iter().any(|row| row.len() != w) {
        return PixelArtVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::with_capacity(pixels.len());
    for row in pixels {
        let mut line = String::with_capacity(w);
        for &cell in row {
            let idx = (cell as usize) % palette.len();
            line.push(palette[idx]);
        }
        lines.push(line);
    }
    PixelArtVerdict::Ok {
        lines,
        width: w as u32,
        height: pixels.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_pixel_art_palette_render")?;

    let pixels = vec![vec![0u8, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
    let palette = vec![' ', '░', '█'];
    println!("smiley: {:?}", render(&pixels, &palette));
    println!("invalid: {:?}", render(&[], &palette));
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
    fn dim_correct() {
        let pixels = vec![vec![0u8; 5]; 3];
        let palette = vec![' '];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { width, height, .. } = v {
            assert_eq!(width, 5);
            assert_eq!(height, 3);
        }
    }

    #[test]
    fn line_count_matches_height() {
        let pixels = vec![vec![0u8]; 3];
        let palette = vec![' '];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn empty_pixels_rejected() {
        let palette = vec![' '];
        assert_eq!(render(&[], &palette), PixelArtVerdict::InvalidConfig);
    }

    #[test]
    fn empty_palette_rejected() {
        let pixels = vec![vec![0u8]];
        assert_eq!(render(&pixels, &[]), PixelArtVerdict::InvalidConfig);
    }

    #[test]
    fn ragged_rows_rejected() {
        let pixels = vec![vec![0u8, 1], vec![0u8]];
        let palette = vec![' '];
        assert_eq!(render(&pixels, &palette), PixelArtVerdict::InvalidConfig);
    }

    #[test]
    fn empty_row_rejected() {
        let pixels: Vec<Vec<u8>> = vec![vec![]];
        let palette = vec![' '];
        assert_eq!(render(&pixels, &palette), PixelArtVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let pixels = vec![vec![0u8]];
        let palette = vec![' '];
        let r1 = render(&pixels, &palette);
        let r2 = render(&pixels, &palette);
        assert_eq!(r1, r2);
    }

    #[test]
    fn palette_index_modulo() {
        let pixels = vec![vec![5u8]];
        let palette = vec!['A', 'B'];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { lines, .. } = v {
            // 5 % 2 = 1 → 'B'
            assert_eq!(lines[0], "B");
        }
    }

    #[test]
    fn line_widths_uniform() {
        let pixels = vec![vec![0u8, 1, 2], vec![1, 2, 0]];
        let palette = vec![' ', '░', '█'];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { lines, .. } = v {
            assert_eq!(lines[0].chars().count(), 3);
            assert_eq!(lines[1].chars().count(), 3);
        }
    }

    #[test]
    fn unicode_palette_supported() {
        let pixels = vec![vec![0u8, 1]];
        let palette = vec!['☃', '☀'];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { lines, .. } = v {
            assert_eq!(lines[0], "☃☀");
        }
    }

    #[test]
    fn single_pixel_works() {
        let pixels = vec![vec![0u8]];
        let palette = vec!['X'];
        let v = render(&pixels, &palette);
        if let PixelArtVerdict::Ok { lines, .. } = v {
            assert_eq!(lines, vec!["X".to_string()]);
        }
    }
}
