//! # TUI Table Sort-Indicator
//!
//! Compute the sort-arrow glyph for a column header given the
//! current sort state. Returns `▲` for ascending, `▼` for descending,
//! and ` ` for unsorted.
//!
//! Demonstrates the **TUI.25** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS NSTableView header sort indicator.
//!
//! Run with: cargo run --example tui_table_sort_indicator
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDir {
    Ascending,
    Descending,
}

#[derive(Debug, PartialEq)]
pub enum SortIndicatorVerdict {
    Pick { glyph: char },
    InvalidColumn,
}

pub fn pick(
    column_name: &str,
    sorted_column: Option<&str>,
    direction: SortDir,
) -> SortIndicatorVerdict {
    if column_name.is_empty() {
        return SortIndicatorVerdict::InvalidColumn;
    }
    match sorted_column {
        Some(name) if name == column_name => {
            let glyph = match direction {
                SortDir::Ascending => '▲',
                SortDir::Descending => '▼',
            };
            SortIndicatorVerdict::Pick { glyph }
        }
        _ => SortIndicatorVerdict::Pick { glyph: ' ' },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_sort_indicator")?;

    println!(
        "active asc: {:?}",
        pick("name", Some("name"), SortDir::Ascending)
    );
    println!(
        "active desc: {:?}",
        pick("name", Some("name"), SortDir::Descending)
    );
    println!(
        "inactive: {:?}",
        pick("name", Some("date"), SortDir::Ascending)
    );
    println!("none: {:?}", pick("name", None, SortDir::Ascending));
    println!("invalid: {:?}", pick("", None, SortDir::Ascending));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn active_ascending_arrow_up() {
        let v = pick("name", Some("name"), SortDir::Ascending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, '▲');
        }
    }

    #[test]
    fn active_descending_arrow_down() {
        let v = pick("name", Some("name"), SortDir::Descending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, '▼');
        }
    }

    #[test]
    fn inactive_column_space() {
        let v = pick("name", Some("date"), SortDir::Ascending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, ' ');
        }
    }

    #[test]
    fn no_sorted_column_space() {
        let v = pick("name", None, SortDir::Ascending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, ' ');
        }
    }

    #[test]
    fn empty_column_invalid() {
        assert_eq!(
            pick("", None, SortDir::Ascending),
            SortIndicatorVerdict::InvalidColumn
        );
    }

    #[test]
    fn case_sensitive_match() {
        let v = pick("Name", Some("name"), SortDir::Ascending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, ' ');
        }
    }

    #[test]
    fn unicode_column_name() {
        let v = pick("café", Some("café"), SortDir::Ascending);
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, '▲');
        }
    }

    #[test]
    fn long_column_name() {
        let v = pick(
            "this_is_a_very_long_column_name",
            Some("this_is_a_very_long_column_name"),
            SortDir::Descending,
        );
        if let SortIndicatorVerdict::Pick { glyph } = v {
            assert_eq!(glyph, '▼');
        }
    }

    #[test]
    fn deterministic() {
        let a = pick("name", Some("name"), SortDir::Ascending);
        let b = pick("name", Some("name"), SortDir::Ascending);
        assert_eq!(a, b);
    }

    #[test]
    fn whitespace_column_invalid() {
        // Empty after no special handling, treated as invalid.
        let v = pick("", Some("any"), SortDir::Ascending);
        assert_eq!(v, SortIndicatorVerdict::InvalidColumn);
    }
}
