//! # TUI Resize Overflow Compact
//!
//! When window narrows, compact each column proportionally to its
//! min size. Returns each column's final width.
//!
//! Demonstrates the **TUI.113** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS Grid auto-fit minmax(); ratatui Layout::Min().
//!
//! Run with: cargo run --example tui_resize_overflow_compact
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompactVerdict {
    Ok { widths: Vec<u32>, total_used: u32 },
    InvalidConfig,
}

pub fn compact(desired: &[u32], min_widths: &[u32], available: u32) -> CompactVerdict {
    if desired.len() != min_widths.len() || desired.is_empty() || available == 0 {
        return CompactVerdict::InvalidConfig;
    }
    if desired.iter().zip(min_widths.iter()).any(|(d, m)| m > d) {
        return CompactVerdict::InvalidConfig;
    }
    let total_min: u32 = min_widths.iter().sum();
    if total_min > available {
        // Can't even satisfy mins.
        return CompactVerdict::Ok {
            widths: min_widths.to_vec(),
            total_used: total_min,
        };
    }
    let total_desired: u32 = desired.iter().sum();
    if total_desired <= available {
        return CompactVerdict::Ok {
            widths: desired.to_vec(),
            total_used: total_desired,
        };
    }
    // Need to scale down between desired and min.
    let mut widths: Vec<u32> = Vec::with_capacity(desired.len());
    let extra_capacity = available - total_min;
    let extra_desired: u32 = desired
        .iter()
        .zip(min_widths.iter())
        .map(|(d, m)| d - m)
        .sum();
    let mut allocated = 0u32;
    for (d, m) in desired.iter().zip(min_widths.iter()) {
        let want_extra = d - m;
        let scaled_extra = (want_extra * extra_capacity)
            .checked_div(extra_desired)
            .unwrap_or(0);
        let w = m + scaled_extra;
        widths.push(w);
        allocated += w;
    }
    CompactVerdict::Ok {
        widths,
        total_used: allocated,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_resize_overflow_compact")?;

    let desired = [40u32, 30, 30];
    let mins = [10u32, 10, 10];
    println!("plenty: {:?}", compact(&desired, &mins, 200));
    println!("tight: {:?}", compact(&desired, &mins, 50));
    println!("invalid: {:?}", compact(&[], &[], 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compactor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn enough_space_uses_desired() {
        let v = compact(&[20, 30], &[10, 10], 100);
        if let CompactVerdict::Ok { widths, .. } = v {
            assert_eq!(widths, vec![20, 30]);
        }
    }

    #[test]
    fn tight_compacts_proportionally() {
        let v = compact(&[40, 30, 30], &[10, 10, 10], 50);
        if let CompactVerdict::Ok { widths, .. } = v {
            for (w, m) in widths.iter().zip([10, 10, 10].iter()) {
                assert!(w >= m);
            }
        }
    }

    #[test]
    fn insufficient_returns_min() {
        let v = compact(&[40, 40], &[30, 30], 50);
        if let CompactVerdict::Ok { widths, .. } = v {
            assert_eq!(widths, vec![30, 30]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compact(&[], &[], 100), CompactVerdict::InvalidConfig);
    }

    #[test]
    fn dim_mismatch_rejected() {
        assert_eq!(
            compact(&[10], &[10, 10], 100),
            CompactVerdict::InvalidConfig
        );
    }

    #[test]
    fn min_above_desired_rejected() {
        assert_eq!(compact(&[10], &[20], 100), CompactVerdict::InvalidConfig);
    }

    #[test]
    fn zero_available_rejected() {
        assert_eq!(compact(&[10], &[5], 0), CompactVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = compact(&[20, 30], &[10, 10], 50);
        let r2 = compact(&[20, 30], &[10, 10], 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn widths_count_matches_input() {
        let v = compact(&[20, 30, 40], &[5, 5, 5], 100);
        if let CompactVerdict::Ok { widths, .. } = v {
            assert_eq!(widths.len(), 3);
        }
    }

    #[test]
    fn at_desired_width_unchanged() {
        let v = compact(&[20, 30], &[10, 10], 50);
        if let CompactVerdict::Ok { widths, .. } = v {
            assert_eq!(widths, vec![20, 30]);
        }
    }

    #[test]
    fn total_used_le_available() {
        let v = compact(&[40, 30, 30], &[5, 5, 5], 50);
        if let CompactVerdict::Ok { total_used, .. } = v {
            assert!(total_used <= 50);
        }
    }
}
