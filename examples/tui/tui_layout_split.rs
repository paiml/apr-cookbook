//! # TUI Flex Layout Split
//!
//! Compute a horizontal flex split: given total_width and weights
//! [w0, w1, ...], return cell widths summing to total_width with
//! remainder distributed deterministically (largest-remainder method).
//!
//! Demonstrates the **TUI.02** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ratatui Constraint::Ratio + largest-remainder rounding.
//!
//! Run with: cargo run --example tui_layout_split
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok { widths: Vec<u32> },
    EmptyWeights,
    AllZeroWeights,
    InvalidWidth,
}

pub fn split(total_width: u32, weights: &[u32]) -> LayoutVerdict {
    if total_width == 0 {
        return LayoutVerdict::InvalidWidth;
    }
    if weights.is_empty() {
        return LayoutVerdict::EmptyWeights;
    }
    let weight_sum: u64 = weights.iter().map(|w| u64::from(*w)).sum();
    if weight_sum == 0 {
        return LayoutVerdict::AllZeroWeights;
    }
    // Compute exact share, floor to u32, track remainder fractions.
    let mut widths = Vec::with_capacity(weights.len());
    let mut remainders: Vec<(usize, u64)> = Vec::with_capacity(weights.len());
    let mut allocated: u64 = 0;
    for (i, w) in weights.iter().enumerate() {
        let scaled = u64::from(total_width) * u64::from(*w);
        let floor = scaled / weight_sum;
        let rem = scaled % weight_sum;
        widths.push(floor as u32);
        allocated += floor;
        remainders.push((i, rem));
    }
    let leftover = u64::from(total_width).saturating_sub(allocated) as usize;
    // Distribute leftover columns to entries with largest remainder.
    remainders.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (i, _) in remainders.iter().take(leftover) {
        widths[*i] += 1;
    }
    LayoutVerdict::Ok { widths }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_layout_split")?;

    println!("equal halves: {:?}", split(80, &[1, 1]));
    println!("uneven: {:?}", split(80, &[1, 2, 1]));
    println!("with rem: {:?}", split(83, &[1, 1, 1]));
    println!("zero weights: {:?}", split(80, &[0, 0]));
    println!("invalid: {:?}", split(0, &[1]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_halves() {
        let v = split(80, &[1, 1]);
        if let LayoutVerdict::Ok { widths } = v {
            assert_eq!(widths, vec![40, 40]);
        }
    }

    #[test]
    fn uneven_proportional() {
        let v = split(80, &[1, 2, 1]);
        if let LayoutVerdict::Ok { widths } = v {
            assert_eq!(widths, vec![20, 40, 20]);
        }
    }

    #[test]
    fn remainder_distributed() {
        let v = split(83, &[1, 1, 1]);
        if let LayoutVerdict::Ok { widths } = v {
            // 83 / 3 = 27 r 2 → largest-remainder gives [28, 28, 27].
            let total: u32 = widths.iter().sum();
            assert_eq!(total, 83);
        }
    }

    #[test]
    fn empty_weights_rejected() {
        assert_eq!(split(80, &[]), LayoutVerdict::EmptyWeights);
    }

    #[test]
    fn zero_total_invalid() {
        assert_eq!(split(0, &[1]), LayoutVerdict::InvalidWidth);
    }

    #[test]
    fn all_zero_weights_rejected() {
        assert_eq!(split(80, &[0, 0]), LayoutVerdict::AllZeroWeights);
    }

    #[test]
    fn single_column_full_width() {
        let v = split(80, &[1]);
        if let LayoutVerdict::Ok { widths } = v {
            assert_eq!(widths, vec![80]);
        }
    }

    #[test]
    fn widths_sum_to_total() {
        for w in [80, 100, 73, 999, 1] {
            let v = split(w, &[3, 1, 4, 1, 5, 9]);
            if let LayoutVerdict::Ok { widths } = v {
                let s: u32 = widths.iter().sum();
                assert_eq!(s, w);
            }
        }
    }

    #[test]
    fn one_dominant_weight() {
        let v = split(100, &[10, 1]);
        if let LayoutVerdict::Ok { widths } = v {
            assert!(widths[0] > widths[1]);
        }
    }

    #[test]
    fn deterministic() {
        let a = split(83, &[1, 2, 3]);
        let b = split(83, &[1, 2, 3]);
        assert_eq!(a, b);
    }
}
