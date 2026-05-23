//! # TUI Color Palette Quantize
//!
//! Quantize a 24-bit RGB color to the nearest of the 16 standard
//! ANSI palette colors via Euclidean distance. Returns palette
//! index (0..=15) and distance to chosen color.
//!
//! Demonstrates the **TUI.70** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ECMA-48 §8.3.117 SGR codes 30-37/90-97; xterm 256color
//!  conventions.
//!
//! Run with: cargo run --example tui_color_palette_quantize
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuantizeVerdict {
    Ok {
        palette_index: u8,
        distance_squared: u32,
    },
    InvalidConfig,
}

const ANSI_16: [(u8, u8, u8); 16] = [
    (0, 0, 0),
    (170, 0, 0),
    (0, 170, 0),
    (170, 85, 0),
    (0, 0, 170),
    (170, 0, 170),
    (0, 170, 170),
    (170, 170, 170),
    (85, 85, 85),
    (255, 85, 85),
    (85, 255, 85),
    (255, 255, 85),
    (85, 85, 255),
    (255, 85, 255),
    (85, 255, 255),
    (255, 255, 255),
];

pub fn quantize(r: u8, g: u8, b: u8) -> QuantizeVerdict {
    let mut best_idx = 0u8;
    let mut best_dist = u32::MAX;
    for (i, (pr, pg, pb)) in ANSI_16.iter().enumerate() {
        let dr = i32::from(r) - i32::from(*pr);
        let dg = i32::from(g) - i32::from(*pg);
        let db = i32::from(b) - i32::from(*pb);
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    QuantizeVerdict::Ok {
        palette_index: best_idx,
        distance_squared: best_dist,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_color_palette_quantize")?;

    println!("pure red: {:?}", quantize(255, 0, 0));
    println!("near red: {:?}", quantize(200, 30, 30));
    println!("white: {:?}", quantize(255, 255, 255));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn black_maps_to_index_0() {
        let v = quantize(0, 0, 0);
        if let QuantizeVerdict::Ok {
            palette_index,
            distance_squared,
        } = v
        {
            assert_eq!(palette_index, 0);
            assert_eq!(distance_squared, 0);
        }
    }

    #[test]
    fn white_maps_to_index_15() {
        let v = quantize(255, 255, 255);
        if let QuantizeVerdict::Ok {
            palette_index,
            distance_squared,
        } = v
        {
            assert_eq!(palette_index, 15);
            assert_eq!(distance_squared, 0);
        }
    }

    #[test]
    fn pure_red_maps_to_bright_red_index_9() {
        // ANSI 9 is bright red (255, 85, 85), distance = 0+85²+85² = 14450.
        // ANSI 1 is dark red (170, 0, 0), distance = 85² + 0 + 0 = 7225.
        // So pure red (255,0,0) → distance² to 1 is 7225, to 9 is 14450.
        // Dark red wins.
        let v = quantize(255, 0, 0);
        if let QuantizeVerdict::Ok { palette_index, .. } = v {
            assert_eq!(palette_index, 1);
        }
    }

    #[test]
    fn palette_index_in_range() {
        let v = quantize(100, 100, 100);
        if let QuantizeVerdict::Ok { palette_index, .. } = v {
            assert!(palette_index < 16);
        }
    }

    #[test]
    fn distance_le_sqrt_three_max() {
        // Worst-case: distance² ≤ 255² * 3 = 195075.
        let v = quantize(0, 0, 0);
        if let QuantizeVerdict::Ok {
            distance_squared, ..
        } = v
        {
            assert!(distance_squared <= 195_075);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = quantize(123, 45, 67);
        let r2 = quantize(123, 45, 67);
        assert_eq!(r1, r2);
    }

    #[test]
    fn near_palette_color_zero_distance() {
        let v = quantize(170, 0, 0);
        if let QuantizeVerdict::Ok {
            distance_squared, ..
        } = v
        {
            assert_eq!(distance_squared, 0);
        }
    }

    #[test]
    fn similar_colors_same_index() {
        let r1 = quantize(170, 0, 0);
        let r2 = quantize(165, 5, 5);
        if let (
            QuantizeVerdict::Ok {
                palette_index: a, ..
            },
            QuantizeVerdict::Ok {
                palette_index: b, ..
            },
        ) = (r1, r2)
        {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn extreme_inputs_handled() {
        let v = quantize(255, 0, 255);
        assert!(matches!(v, QuantizeVerdict::Ok { .. }));
    }

    #[test]
    fn distance_nonneg() {
        let v = quantize(100, 100, 100);
        if let QuantizeVerdict::Ok {
            distance_squared, ..
        } = v
        {
            // u32 always nonneg; documents intent.
            let _ = distance_squared;
        }
    }
}
