//! # TUI ASCII Histogram Render
//!
//! Render bar-chart bins as ASCII bars proportional to bin counts.
//! Returns rendered lines (label + bar + count) and the max-bar
//! width used.
//!
//! Demonstrates the **TUI.164** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: gnuplot ASCII terminal output; cli-table2 bar rendering.
//!
//! Run with: cargo run --example tui_ascii_histogram_render
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HistogramVerdict {
    Ok {
        lines: Vec<String>,
        max_bar_width: u32,
    },
    InvalidConfig,
}

pub fn render(bins: &[(&str, u32)], max_bar: u32) -> HistogramVerdict {
    if bins.is_empty() || !(5..=200).contains(&max_bar) {
        return HistogramVerdict::InvalidConfig;
    }
    let max_count = bins.iter().map(|(_, c)| *c).max().unwrap_or(1).max(1);
    let label_w = bins
        .iter()
        .map(|(l, _)| l.chars().count())
        .max()
        .unwrap_or(0);
    let mut lines: Vec<String> = Vec::with_capacity(bins.len());
    for (label, count) in bins {
        let bar_len = ((*count as f64 / max_count as f64) * max_bar as f64) as usize;
        let bar = "█".repeat(bar_len);
        let line = format!(
            "{label:<lw$} {bar} {count}",
            label = label,
            lw = label_w,
            bar = bar,
            count = count
        );
        lines.push(line);
    }
    HistogramVerdict::Ok {
        lines,
        max_bar_width: max_bar,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_ascii_histogram_render")?;

    let bins = [("a", 5), ("b", 10), ("c", 3)];
    println!("histogram: {:?}", render(&bins, 20));
    println!("invalid: {:?}", render(&[], 20));
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
        assert_eq!(render(&[], 20), HistogramVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_small_bar() {
        assert_eq!(render(&[("a", 1)], 2), HistogramVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_large_bar() {
        assert_eq!(render(&[("a", 1)], 1000), HistogramVerdict::InvalidConfig);
    }

    #[test]
    fn line_count_matches() {
        let v = render(&[("a", 1), ("b", 2)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 2);
        }
    }

    #[test]
    fn max_count_full_bar() {
        let v = render(&[("a", 10), ("b", 20)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            // "b" has the max count → its bar should span max_bar = 20 blocks
            let b_count_blocks = lines[1].matches('█').count();
            assert_eq!(b_count_blocks, 20);
        }
    }

    #[test]
    fn label_left_aligned() {
        let v = render(&[("apple", 5), ("a", 5)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            // Both labels padded to width 5 (longest)
            assert!(lines[0].starts_with("apple "));
            assert!(lines[1].starts_with("a    "));
        }
    }

    #[test]
    fn count_appears_in_line() {
        let v = render(&[("a", 42)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            assert!(lines[0].ends_with("42"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&[("a", 5)], 20);
        let r2 = render(&[("a", 5)], 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn max_bar_width_returned() {
        let v = render(&[("a", 5)], 25);
        if let HistogramVerdict::Ok { max_bar_width, .. } = v {
            assert_eq!(max_bar_width, 25);
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render(&[("café", 5)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("café"));
        }
    }

    #[test]
    fn many_bins_handled() {
        let bins: Vec<(&str, u32)> = (0..30).map(|_| ("b", 5)).collect();
        let v = render(&bins, 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 30);
        }
    }

    #[test]
    fn zero_count_empty_bar() {
        let v = render(&[("a", 0), ("b", 10)], 20);
        if let HistogramVerdict::Ok { lines, .. } = v {
            assert!(!lines[0].contains('█'));
        }
    }
}
