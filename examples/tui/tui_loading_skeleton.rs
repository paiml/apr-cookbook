//! # TUI Loading Skeleton Generator
//!
//! Produce skeleton placeholder rows for a loading state. Each row is
//! a `width × 1` box with a moving "shimmer" position based on the
//! tick count, simulating the iOS / web skeleton-shimmer pattern.
//!
//! Demonstrates the **TUI.33** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: shimmer skeleton pattern (Facebook React UX 2017).
//!
//! Run with: cargo run --example tui_loading_skeleton
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkeletonVerdict {
    Ok { rows: Vec<String>, shimmer_col: u32 },
    InvalidConfig,
}

pub fn generate(rows: u32, width: u32, tick: u64) -> SkeletonVerdict {
    if rows == 0 || width == 0 {
        return SkeletonVerdict::InvalidConfig;
    }
    let shimmer_col = (tick % u64::from(width)) as u32;
    let mut output = Vec::with_capacity(rows as usize);
    for _ in 0..rows {
        let mut line = String::with_capacity(width as usize);
        for c in 0..width {
            let ch = if c == shimmer_col { '▒' } else { '░' };
            line.push(ch);
        }
        output.push(line);
    }
    SkeletonVerdict::Ok {
        rows: output,
        shimmer_col,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_loading_skeleton")?;

    println!("typical: {:?}", generate(3, 20, 0));
    println!("tick advances: {:?}", generate(3, 20, 5));
    println!("wrap: {:?}", generate(3, 20, 100));
    println!("invalid: {:?}", generate(0, 20, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn row_count_matches_input() {
        let v = generate(5, 10, 0);
        if let SkeletonVerdict::Ok { rows, .. } = v {
            assert_eq!(rows.len(), 5);
        }
    }

    #[test]
    fn each_row_correct_width() {
        let v = generate(3, 15, 0);
        if let SkeletonVerdict::Ok { rows, .. } = v {
            for row in rows {
                assert_eq!(row.chars().count(), 15);
            }
        }
    }

    #[test]
    fn shimmer_advances_with_tick() {
        let v0 = generate(1, 20, 0);
        let v1 = generate(1, 20, 1);
        if let (
            SkeletonVerdict::Ok {
                shimmer_col: c0, ..
            },
            SkeletonVerdict::Ok {
                shimmer_col: c1, ..
            },
        ) = (v0, v1)
        {
            assert_eq!(c1, c0 + 1);
        }
    }

    #[test]
    fn shimmer_wraps_at_width() {
        let v = generate(1, 20, 100);
        if let SkeletonVerdict::Ok { shimmer_col, .. } = v {
            assert_eq!(shimmer_col, 0); // 100 % 20.
        }
    }

    #[test]
    fn invalid_zero_rows() {
        assert_eq!(generate(0, 10, 0), SkeletonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(generate(3, 0, 0), SkeletonVerdict::InvalidConfig);
    }

    #[test]
    fn shimmer_in_bounds() {
        for tick in [0, 5, 100, 9999] {
            let v = generate(1, 20, tick);
            if let SkeletonVerdict::Ok { shimmer_col, .. } = v {
                assert!(shimmer_col < 20);
            }
        }
    }

    #[test]
    fn shimmer_char_in_row() {
        let v = generate(1, 10, 3);
        if let SkeletonVerdict::Ok { rows, shimmer_col } = v {
            assert_eq!(rows[0].chars().nth(shimmer_col as usize), Some('▒'));
        }
    }

    #[test]
    fn other_chars_are_filler() {
        let v = generate(1, 10, 0);
        if let SkeletonVerdict::Ok { rows, .. } = v {
            // Position 0 is shimmer ▒; rest should be ░.
            for c in rows[0].chars().skip(1) {
                assert_eq!(c, '░');
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = generate(3, 20, 5);
        let b = generate(3, 20, 5);
        assert_eq!(a, b);
    }
}
