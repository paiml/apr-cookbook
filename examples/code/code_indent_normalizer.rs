//! # Code Indent Normalizer
//!
//! Convert tabs ↔ spaces with auto-detected tab-width. Detection:
//! examine first 100 indented lines; mode of (leading_spaces /
//! suspected_width) wins. Common widths: 2, 4, 8. This recipe builds
//! the detector + the converter.
//!
//! Demonstrates the **CODE.9** recipe for PMAT-125 (code coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PEP 8 §indentation; GNU coding standards §1.
//!
//! Run with: cargo run --example code_indent_normalizer
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SAMPLE_SIZE: usize = 100;

#[derive(Debug, PartialEq, Eq)]
pub enum IndentStyle {
    Tabs,
    Spaces { width: u8 },
    Mixed,
    Unknown,
}

pub fn detect(source: &str) -> IndentStyle {
    let mut tab_count = 0u32;
    let mut space_widths: [u32; 9] = [0; 9]; // index = width 1..=8
    for line in source.lines().take(SAMPLE_SIZE) {
        if line.trim_start() == line {
            continue;
        }
        let leading_tabs = line.bytes().take_while(|&b| b == b'\t').count();
        let leading_spaces = line.bytes().take_while(|&b| b == b' ').count();
        if leading_tabs > 0 && leading_spaces == 0 {
            tab_count += 1;
        } else if leading_spaces > 0 && leading_tabs == 0 {
            for w in 2u8..=8 {
                if leading_spaces % usize::from(w) == 0 {
                    space_widths[usize::from(w)] += 1;
                }
            }
        }
    }
    if tab_count > 0 && space_widths.iter().all(|&c| c == 0) {
        return IndentStyle::Tabs;
    }
    if tab_count == 0 {
        let max = space_widths.iter().enumerate().max_by_key(|(_, &c)| c);
        if let Some((idx, &count)) = max {
            if count > 0 {
                return IndentStyle::Spaces { width: idx as u8 };
            }
        }
        return IndentStyle::Unknown;
    }
    IndentStyle::Mixed
}

pub fn convert_tabs_to_spaces(source: &str, width: u8) -> String {
    if width == 0 {
        return source.to_string();
    }
    let pad = " ".repeat(usize::from(width));
    source.replace('\t', &pad)
}

pub fn convert_spaces_to_tabs(source: &str, width: u8) -> String {
    if width == 0 {
        return source.to_string();
    }
    let pad = " ".repeat(usize::from(width));
    let mut out = String::with_capacity(source.len());
    for line in source.split_inclusive('\n') {
        let leading_spaces = line.bytes().take_while(|&b| b == b' ').count();
        if leading_spaces == 0 {
            out.push_str(line);
            continue;
        }
        let tabs = leading_spaces / usize::from(width);
        let remainder = leading_spaces % usize::from(width);
        out.push_str(&"\t".repeat(tabs));
        out.push_str(&pad[..remainder]);
        out.push_str(&line[leading_spaces..]);
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_indent_normalizer")?;

    let two_space = "fn x() {\n  let y = 1;\n  if y > 0 {\n    println!(\"hi\");\n  }\n}\n";
    let tab_indent = "fn x() {\n\tlet y = 1;\n\tif y > 0 {\n\t\tprintln!(\"hi\");\n\t}\n}\n";

    println!("2-space detect: {:?}", detect(two_space));
    println!("tab detect:    {:?}", detect(tab_indent));

    let converted = convert_tabs_to_spaces(tab_indent, 4);
    println!("→ 4 spaces:\n{converted}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn detect_tabs_only() {
        let s = "fn x() {\n\tlet y = 1;\n\treturn y;\n}\n";
        assert_eq!(detect(s), IndentStyle::Tabs);
    }

    #[test]
    fn detect_4_space_indentation() {
        let s = "fn x() {\n    let y = 1;\n    return y;\n}\n";
        // 4 spaces is a multiple of both 2 and 4; 4 wins on mode (highest count).
        if let IndentStyle::Spaces { width } = detect(s) {
            assert!(matches!(width, 2 | 4));
        } else {
            panic!("expected Spaces");
        }
    }

    #[test]
    fn detect_2_space_indentation() {
        let s = "fn x() {\n  let y = 1;\n  if y > 0 {\n    return y;\n  }\n}\n";
        if let IndentStyle::Spaces { width } = detect(s) {
            assert_eq!(width, 2);
        }
    }

    #[test]
    fn detect_mixed_indentation() {
        let s = "fn x() {\n\tlet a = 1;\n  let b = 2;\n}\n";
        assert_eq!(detect(s), IndentStyle::Mixed);
    }

    #[test]
    fn detect_unknown_no_indent() {
        let s = "fn x() {}\n";
        assert_eq!(detect(s), IndentStyle::Unknown);
    }

    #[test]
    fn convert_tabs_to_spaces_works() {
        let s = "\tfoo\n\t\tbar";
        let r = convert_tabs_to_spaces(s, 2);
        assert_eq!(r, "  foo\n    bar");
    }

    #[test]
    fn convert_spaces_to_tabs_works() {
        let s = "    foo\n        bar";
        let r = convert_spaces_to_tabs(s, 4);
        assert_eq!(r, "\tfoo\n\t\tbar");
    }

    #[test]
    fn convert_zero_width_returns_input() {
        assert_eq!(convert_tabs_to_spaces("\tfoo", 0), "\tfoo");
        assert_eq!(convert_spaces_to_tabs("  foo", 0), "  foo");
    }

    #[test]
    fn round_trip_preserves_content() {
        let s = "    let x = 1;\n        if x > 0 { return; }\n";
        let to_tabs = convert_spaces_to_tabs(s, 4);
        let back = convert_tabs_to_spaces(&to_tabs, 4);
        assert_eq!(back, s);
    }

    #[test]
    fn convert_handles_partial_indent() {
        // 5 spaces with width 4 → 1 tab + 1 space.
        let s = "     foo";
        let r = convert_spaces_to_tabs(s, 4);
        assert_eq!(r, "\t foo");
    }
}
