//! # TUI Minimap Render
//!
//! Bucket an N-line file into a `height`-row minimap. Each row's
//! density = (lines mapped to it / file_lines) and is rendered as a
//! 0..=8 unicode block-character.
//!
//! Demonstrates the **TUI.66** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sublime Text minimap; VS Code editor.minimap.
//!
//! Run with: cargo run --example tui_minimap_render
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MinimapVerdict {
    Ok {
        glyphs: Vec<char>,
        per_bucket_count: Vec<u32>,
    },
    InvalidConfig,
}

pub fn render(file_lines: u32, height: u32) -> MinimapVerdict {
    if file_lines == 0 || height == 0 || height > file_lines {
        return MinimapVerdict::InvalidConfig;
    }
    let mut per_bucket: Vec<u32> = vec![0; height as usize];
    for line in 0..file_lines {
        let bucket = (u64::from(line) * u64::from(height) / u64::from(file_lines)) as usize;
        let bucket = bucket.min(height as usize - 1);
        per_bucket[bucket] += 1;
    }
    let max_in_bucket = per_bucket.iter().max().copied().unwrap_or(1);
    let glyphs: Vec<char> = per_bucket
        .iter()
        .map(|&c| {
            let level = (u64::from(c) * 8 / u64::from(max_in_bucket.max(1))) as u8;
            block_char(level)
        })
        .collect();
    MinimapVerdict::Ok {
        glyphs,
        per_bucket_count: per_bucket,
    }
}

fn block_char(level: u8) -> char {
    match level {
        0 => ' ',
        1 => '▁',
        2 => '▂',
        3 => '▃',
        4 => '▄',
        5 => '▅',
        6 => '▆',
        7 => '▇',
        _ => '█',
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_minimap_render")?;

    println!("uniform: {:?}", render(100, 10));
    println!("tiny: {:?}", render(5, 5));
    println!("invalid: {:?}", render(10, 100));
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
    fn uniform_distribution_consistent_levels() {
        let v = render(100, 10);
        if let MinimapVerdict::Ok {
            per_bucket_count, ..
        } = v
        {
            // Each bucket should contain ~10 lines.
            for c in &per_bucket_count {
                assert!(*c >= 9 && *c <= 11);
            }
        }
    }

    #[test]
    fn glyph_count_matches_height() {
        let v = render(100, 7);
        if let MinimapVerdict::Ok { glyphs, .. } = v {
            assert_eq!(glyphs.len(), 7);
        }
    }

    #[test]
    fn invalid_zero_lines() {
        assert_eq!(render(0, 10), MinimapVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_height() {
        assert_eq!(render(100, 0), MinimapVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_height_gt_lines() {
        assert_eq!(render(10, 100), MinimapVerdict::InvalidConfig);
    }

    #[test]
    fn one_to_one_each_bucket_one_line() {
        let v = render(5, 5);
        if let MinimapVerdict::Ok {
            per_bucket_count, ..
        } = v
        {
            assert!(per_bucket_count.iter().all(|c| *c == 1));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(100, 10);
        let r2 = render(100, 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn total_count_equals_file_lines() {
        let v = render(100, 8);
        if let MinimapVerdict::Ok {
            per_bucket_count, ..
        } = v
        {
            let total: u32 = per_bucket_count.iter().sum();
            assert_eq!(total, 100);
        }
    }

    #[test]
    fn glyphs_are_block_chars() {
        let v = render(100, 10);
        if let MinimapVerdict::Ok { glyphs, .. } = v {
            for g in &glyphs {
                assert!(matches!(
                    *g,
                    ' ' | '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'
                ));
            }
        }
    }

    #[test]
    fn single_height_one_bucket() {
        let v = render(100, 1);
        if let MinimapVerdict::Ok {
            glyphs,
            per_bucket_count,
        } = v
        {
            assert_eq!(glyphs.len(), 1);
            assert_eq!(per_bucket_count[0], 100);
        }
    }

    #[test]
    fn larger_file_more_per_bucket() {
        let small = render(50, 5);
        let big = render(500, 5);
        if let (
            MinimapVerdict::Ok {
                per_bucket_count: s,
                ..
            },
            MinimapVerdict::Ok {
                per_bucket_count: b,
                ..
            },
        ) = (small, big)
        {
            assert!(b[0] > s[0]);
        }
    }
}
