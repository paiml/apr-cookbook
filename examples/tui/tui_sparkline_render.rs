//! # TUI Sparkline Render
//!
//! Render a sequence of values as Unicode block-characters (▁▂▃▄▅▆▇█)
//! sized proportionally to value range. Returns the sparkline string
//! and the (min, max) range used.
//!
//! Demonstrates the **TUI.163** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tufte, Beautiful Evidence (2006) on intense word-sized
//!  graphics; sparkline.js library convention.
//!
//! Run with: cargo run --example tui_sparkline_render
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SparklineVerdict {
    Ok {
        rendered: String,
        min: u32,
        max: u32,
    },
    InvalidConfig,
}

pub fn render(values: &[u32]) -> SparklineVerdict {
    if values.is_empty() {
        return SparklineVerdict::InvalidConfig;
    }
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = *values.iter().min().unwrap_or(&0);
    let max = *values.iter().max().unwrap_or(&0);
    let mut rendered = String::with_capacity(values.len() * 3);
    for v in values {
        let idx = if max == min {
            0
        } else {
            ((*v - min) as f64 / (max - min) as f64 * 7.999) as usize
        };
        rendered.push(bars[idx]);
    }
    SparklineVerdict::Ok { rendered, min, max }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_sparkline_render")?;

    println!("ramp: {:?}", render(&[1, 2, 3, 4, 5, 6, 7, 8]));
    println!("flat: {:?}", render(&[5, 5, 5, 5]));
    println!("invalid: {:?}", render(&[]));
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
        assert_eq!(render(&[]), SparklineVerdict::InvalidConfig);
    }

    #[test]
    fn min_max_correct() {
        let v = render(&[3, 1, 4, 1, 5, 9, 2, 6]);
        if let SparklineVerdict::Ok { min, max, .. } = v {
            assert_eq!(min, 1);
            assert_eq!(max, 9);
        }
    }

    #[test]
    fn rendered_length_matches() {
        let v = render(&[1, 2, 3, 4, 5]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.chars().count(), 5);
        }
    }

    #[test]
    fn ramp_increases_density() {
        let v = render(&[1, 8]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            let chars: Vec<char> = rendered.chars().collect();
            assert_eq!(chars[0], '▁');
            assert_eq!(chars[1], '█');
        }
    }

    #[test]
    fn flat_all_same_bar() {
        let v = render(&[5, 5, 5]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            let chars: Vec<char> = rendered.chars().collect();
            assert_eq!(chars[0], chars[1]);
            assert_eq!(chars[1], chars[2]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&[1, 2, 3]);
        let r2 = render(&[1, 2, 3]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_value_handled() {
        let v = render(&[42]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.chars().count(), 1);
        }
    }

    #[test]
    fn min_value_bottom_bar() {
        let v = render(&[1, 100]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('▁'));
        }
    }

    #[test]
    fn max_value_top_bar() {
        let v = render(&[1, 100]);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            assert!(rendered.ends_with('█'));
        }
    }

    #[test]
    fn many_values_handled() {
        let values: Vec<u32> = (0..100).collect();
        let v = render(&values);
        if let SparklineVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.chars().count(), 100);
        }
    }

    #[test]
    fn zero_min_zero_max_no_panic() {
        let v = render(&[0, 0]);
        assert!(matches!(v, SparklineVerdict::Ok { .. }));
    }

    #[test]
    fn high_max_handled() {
        let v = render(&[1, 1_000_000]);
        if let SparklineVerdict::Ok { max, .. } = v {
            assert_eq!(max, 1_000_000);
        }
    }
}
