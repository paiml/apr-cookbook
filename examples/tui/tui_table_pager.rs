//! # TUI Table Pager
//!
//! Compute pagination state for a scrollable table: given total_rows
//! and page_size, return current page index, slice [start, end), and
//! total page count. No terminal IO.
//!
//! Demonstrates the **TUI.04** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ratatui Table widget pagination patterns.
//!
//! Run with: cargo run --example tui_table_pager
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PagerVerdict {
    Ok {
        page_index: u32,
        page_count: u32,
        start: u32,
        end: u32,
    },
    EmptyTable,
    InvalidPageSize,
}

pub fn paginate(total_rows: u32, page_size: u32, requested_page: u32) -> PagerVerdict {
    if total_rows == 0 {
        return PagerVerdict::EmptyTable;
    }
    if page_size == 0 {
        return PagerVerdict::InvalidPageSize;
    }
    let page_count = total_rows.div_ceil(page_size);
    let page_index = requested_page.min(page_count - 1);
    let start = page_index * page_size;
    let end = (start + page_size).min(total_rows);
    PagerVerdict::Ok {
        page_index,
        page_count,
        start,
        end,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_pager")?;

    println!("first page: {:?}", paginate(100, 25, 0));
    println!("middle: {:?}", paginate(100, 25, 2));
    println!("last partial: {:?}", paginate(110, 25, 4));
    println!("over-requested: {:?}", paginate(100, 25, 99));
    println!("empty: {:?}", paginate(0, 25, 0));
    println!("zero page size: {:?}", paginate(100, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pager_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_page_zero_to_25() {
        let v = paginate(100, 25, 0);
        if let PagerVerdict::Ok { start, end, .. } = v {
            assert_eq!(start, 0);
            assert_eq!(end, 25);
        }
    }

    #[test]
    fn page_count_correct() {
        let v = paginate(100, 25, 0);
        if let PagerVerdict::Ok { page_count, .. } = v {
            assert_eq!(page_count, 4);
        }
    }

    #[test]
    fn last_partial_page_clamps() {
        let v = paginate(110, 25, 4);
        if let PagerVerdict::Ok { start, end, .. } = v {
            assert_eq!(start, 100);
            assert_eq!(end, 110);
        }
    }

    #[test]
    fn requested_page_clamped_max() {
        let v = paginate(100, 25, 99);
        if let PagerVerdict::Ok { page_index, .. } = v {
            assert_eq!(page_index, 3);
        }
    }

    #[test]
    fn empty_table_rejected() {
        assert_eq!(paginate(0, 25, 0), PagerVerdict::EmptyTable);
    }

    #[test]
    fn zero_page_size_rejected() {
        assert_eq!(paginate(100, 0, 0), PagerVerdict::InvalidPageSize);
    }

    #[test]
    fn single_row_single_page() {
        let v = paginate(1, 25, 0);
        if let PagerVerdict::Ok {
            page_count,
            start,
            end,
            ..
        } = v
        {
            assert_eq!(page_count, 1);
            assert_eq!(start, 0);
            assert_eq!(end, 1);
        }
    }

    #[test]
    fn partial_first_page_works() {
        let v = paginate(10, 25, 0);
        if let PagerVerdict::Ok {
            page_count, end, ..
        } = v
        {
            assert_eq!(page_count, 1);
            assert_eq!(end, 10);
        }
    }

    #[test]
    fn page_index_in_range() {
        for total in [10, 25, 100, 1000] {
            for size in [1, 5, 25, 100] {
                for req in [0, 5, 100] {
                    let v = paginate(total, size, req);
                    if let PagerVerdict::Ok {
                        page_index,
                        page_count,
                        ..
                    } = v
                    {
                        assert!(page_index < page_count);
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = paginate(100, 25, 2);
        let b = paginate(100, 25, 2);
        assert_eq!(a, b);
    }
}
