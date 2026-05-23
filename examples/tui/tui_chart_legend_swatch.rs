//! # TUI Chart Legend Swatch
//!
//! Render a chart legend: colored swatch + label per series. Returns
//! rendered lines and total width (for layout planning).
//!
//! Demonstrates the **TUI.108** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: matplotlib legend convention; D3.js color scales.
//!
//! Run with: cargo run --example tui_chart_legend_swatch
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LegendVerdict {
    Ok {
        rendered: Vec<String>,
        total_width: u32,
    },
    InvalidConfig,
}

pub fn render(series: &[(&str, char)]) -> LegendVerdict {
    if series.is_empty() {
        return LegendVerdict::InvalidConfig;
    }
    let mut rendered: Vec<String> = Vec::with_capacity(series.len());
    let mut max_width: u32 = 0;
    for (label, swatch_char) in series {
        // Swatch: "■" or other glyph followed by space + label.
        let line = format!("{swatch_char} {label}");
        let w = line.chars().count() as u32;
        if w > max_width {
            max_width = w;
        }
        rendered.push(line);
    }
    LegendVerdict::Ok {
        rendered,
        total_width: max_width,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_chart_legend_swatch")?;

    let series = [("CPU", '■'), ("Memory", '●'), ("Network", '▲')];
    println!("legend: {:?}", render(&series));
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
    fn count_matches_series() {
        let series = [("CPU", '■'), ("Memory", '●')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.len(), 2);
        }
    }

    #[test]
    fn swatch_and_label_present() {
        let series = [("CPU", '■')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert!(rendered[0].contains('■'));
            assert!(rendered[0].contains("CPU"));
        }
    }

    #[test]
    fn empty_series_rejected() {
        assert_eq!(render(&[]), LegendVerdict::InvalidConfig);
    }

    #[test]
    fn total_width_is_max_label_width() {
        let series = [("X", '■'), ("LongerName", '●')];
        let v = render(&series);
        if let LegendVerdict::Ok { total_width, .. } = v {
            assert_eq!(total_width, "● LongerName".chars().count() as u32);
        }
    }

    #[test]
    fn deterministic() {
        let series = [("a", '■')];
        let r1 = render(&series);
        let r2 = render(&series);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_label_supported() {
        let series = [("café", '■')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert!(rendered[0].contains("café"));
        }
    }

    #[test]
    fn swatch_per_series_distinct_supported() {
        let series = [("a", '■'), ("b", '●'), ("c", '▲')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert!(rendered[0].contains('■'));
            assert!(rendered[1].contains('●'));
            assert!(rendered[2].contains('▲'));
        }
    }

    #[test]
    fn single_series_works() {
        let series = [("only", '■')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.len(), 1);
        }
    }

    #[test]
    fn rendered_lines_format_correct() {
        let series = [("X", '■')];
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered[0], "■ X");
        }
    }

    #[test]
    fn many_series_handled() {
        let series: Vec<(&str, char)> = (0..10).map(|_| ("s", '■')).collect();
        let v = render(&series);
        if let LegendVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.len(), 10);
        }
    }

    #[test]
    fn total_width_nonneg() {
        let series = [("X", '■')];
        let v = render(&series);
        if let LegendVerdict::Ok { total_width, .. } = v {
            assert!(total_width > 0);
        }
    }
}
