//! # TUI Table Zebra Stripe
//!
//! Compute zebra-stripe (alternating background) for table rows.
//! Returns each row's bg color tag — `"normal"` or `"alt"` — given
//! row index and stripe period (default 1).
//!
//! Demonstrates the **TUI.69** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ANSI table styling conventions; ratatui Table widget.
//!
//! Run with: cargo run --example tui_table_zebra_stripe
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ZebraVerdict {
    Ok { tags: Vec<String> },
    InvalidConfig,
}

pub fn stripe(row_count: u32, period: u32) -> ZebraVerdict {
    if row_count == 0 || period == 0 {
        return ZebraVerdict::InvalidConfig;
    }
    let mut tags: Vec<String> = Vec::with_capacity(row_count as usize);
    for i in 0..row_count {
        let band = (i / period) % 2;
        let tag = if band == 0 { "normal" } else { "alt" };
        tags.push(tag.to_string());
    }
    ZebraVerdict::Ok { tags }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_zebra_stripe")?;

    println!("period 1: {:?}", stripe(6, 1));
    println!("period 2: {:?}", stripe(8, 2));
    println!("invalid: {:?}", stripe(0, 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn striper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn period_one_alternates_each_row() {
        let v = stripe(4, 1);
        if let ZebraVerdict::Ok { tags } = v {
            assert_eq!(
                tags,
                vec![
                    "normal".to_string(),
                    "alt".to_string(),
                    "normal".to_string(),
                    "alt".to_string()
                ]
            );
        }
    }

    #[test]
    fn period_two_groups_two() {
        let v = stripe(6, 2);
        if let ZebraVerdict::Ok { tags } = v {
            assert_eq!(tags[0], "normal");
            assert_eq!(tags[1], "normal");
            assert_eq!(tags[2], "alt");
            assert_eq!(tags[3], "alt");
            assert_eq!(tags[4], "normal");
            assert_eq!(tags[5], "normal");
        }
    }

    #[test]
    fn zero_rows_rejected() {
        assert_eq!(stripe(0, 1), ZebraVerdict::InvalidConfig);
    }

    #[test]
    fn zero_period_rejected() {
        assert_eq!(stripe(5, 0), ZebraVerdict::InvalidConfig);
    }

    #[test]
    fn first_row_always_normal() {
        let v = stripe(1, 1);
        if let ZebraVerdict::Ok { tags } = v {
            assert_eq!(tags[0], "normal");
        }
    }

    #[test]
    fn count_matches_row_count() {
        let v = stripe(10, 3);
        if let ZebraVerdict::Ok { tags } = v {
            assert_eq!(tags.len(), 10);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = stripe(8, 2);
        let r2 = stripe(8, 2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn only_two_distinct_tags() {
        let v = stripe(20, 1);
        if let ZebraVerdict::Ok { tags } = v {
            for t in &tags {
                assert!(t == "normal" || t == "alt");
            }
        }
    }

    #[test]
    fn large_period_first_band_only() {
        let v = stripe(5, 100);
        if let ZebraVerdict::Ok { tags } = v {
            for t in &tags {
                assert_eq!(t, "normal");
            }
        }
    }

    #[test]
    fn period_three_works() {
        let v = stripe(6, 3);
        if let ZebraVerdict::Ok { tags } = v {
            assert_eq!(tags[0], "normal");
            assert_eq!(tags[2], "normal");
            assert_eq!(tags[3], "alt");
            assert_eq!(tags[5], "alt");
        }
    }

    #[test]
    fn alternates_have_equal_count_at_even_rows() {
        let v = stripe(10, 1);
        if let ZebraVerdict::Ok { tags } = v {
            let normals = tags.iter().filter(|t| **t == "normal").count();
            let alts = tags.iter().filter(|t| **t == "alt").count();
            assert_eq!(normals, alts);
        }
    }
}
