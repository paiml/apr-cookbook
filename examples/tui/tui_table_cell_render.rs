//! # TUI Table Cell Renderer
//!
//! Render a table cell with column-type-aware formatting:
//!   Integer: right-align, comma-thousands
//!   Float: 2 decimal places, right-align
//!   Bool: ✓ / ✗ glyph centered
//!   Text: left-align, truncate with ellipsis
//!
//! Demonstrates the **TUI.43** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: spreadsheet column type formatting (Excel / Numbers).
//!
//! Run with: cargo run --example tui_table_cell_render
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Float,
    Bool,
    Text,
}

#[derive(Debug, PartialEq)]
pub enum CellVerdict {
    Ok { rendered: String },
    InvalidWidth,
    ParseError { reason: &'static str },
}

pub fn render(value: &str, col_type: ColumnType, width: usize) -> CellVerdict {
    if width == 0 {
        return CellVerdict::InvalidWidth;
    }
    let formatted = match col_type {
        ColumnType::Integer => {
            let Ok(n) = value.parse::<i64>() else {
                return CellVerdict::ParseError {
                    reason: "not an integer",
                };
            };
            with_commas(n)
        }
        ColumnType::Float => {
            let Ok(f) = value.parse::<f64>() else {
                return CellVerdict::ParseError {
                    reason: "not a float",
                };
            };
            format!("{f:.2}")
        }
        ColumnType::Bool => {
            let trimmed = value.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "true" | "1" | "yes" => "✓".to_string(),
                "false" | "0" | "no" => "✗".to_string(),
                _ => return CellVerdict::ParseError { reason: "bad bool" },
            }
        }
        ColumnType::Text => value.to_string(),
    };
    let n = formatted.chars().count();
    if n > width {
        let mut truncated: String = formatted.chars().take(width.saturating_sub(1)).collect();
        truncated.push('…');
        return CellVerdict::Ok {
            rendered: truncated,
        };
    }
    let pad = width - n;
    let rendered = match col_type {
        ColumnType::Integer | ColumnType::Float => format!("{}{formatted}", " ".repeat(pad)),
        ColumnType::Bool => {
            let l = pad / 2;
            let r = pad - l;
            format!("{}{formatted}{}", " ".repeat(l), " ".repeat(r))
        }
        ColumnType::Text => format!("{formatted}{}", " ".repeat(pad)),
    };
    CellVerdict::Ok { rendered }
}

fn with_commas(n: i64) -> String {
    let abs = n.unsigned_abs();
    let digits = abs.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_cell_render")?;

    println!("integer: {:?}", render("1234567", ColumnType::Integer, 12));
    println!("float: {:?}", render("3.14159", ColumnType::Float, 8));
    println!("bool: {:?}", render("true", ColumnType::Bool, 5));
    println!("text: {:?}", render("hello world", ColumnType::Text, 8));
    println!("parse err: {:?}", render("xyz", ColumnType::Integer, 5));
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
    fn integer_with_commas() {
        let v = render("1234567", ColumnType::Integer, 12);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.contains("1,234,567"));
        }
    }

    #[test]
    fn float_two_decimals() {
        let v = render("3.14159", ColumnType::Float, 8);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.contains("3.14"));
        }
    }

    #[test]
    fn bool_true_glyph() {
        let v = render("true", ColumnType::Bool, 5);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.contains('✓'));
        }
    }

    #[test]
    fn bool_false_glyph() {
        let v = render("false", ColumnType::Bool, 5);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.contains('✗'));
        }
    }

    #[test]
    fn text_left_aligned() {
        let v = render("hello", ColumnType::Text, 10);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.starts_with("hello"));
        }
    }

    #[test]
    fn over_width_truncated() {
        let v = render("hello world", ColumnType::Text, 8);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.ends_with('…'));
        }
    }

    #[test]
    fn integer_parse_error() {
        let v = render("abc", ColumnType::Integer, 5);
        assert!(matches!(v, CellVerdict::ParseError { .. }));
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(render("x", ColumnType::Text, 0), CellVerdict::InvalidWidth);
    }

    #[test]
    fn negative_integer() {
        let v = render("-1234", ColumnType::Integer, 8);
        if let CellVerdict::Ok { rendered } = v {
            assert!(rendered.contains("-1,234"));
        }
    }

    #[test]
    fn deterministic() {
        let a = render("1234567", ColumnType::Integer, 12);
        let b = render("1234567", ColumnType::Integer, 12);
        assert_eq!(a, b);
    }
}
