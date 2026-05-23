//! # TUI Breadcrumb Path Renderer
//!
//! Render breadcrumb segments to a single-line string with separator
//! between segments. Truncate from the front when the rendered string
//! would exceed `width`, prepending `…` to indicate hidden ancestors.
//!
//! Demonstrates the **TUI.07** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Finder + Windows Explorer breadcrumb conventions.
//!
//! Run with: cargo run --example tui_breadcrumb_path
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BreadcrumbVerdict {
    Ok { rendered: String, hidden: u32 },
    EmptyPath,
    InvalidWidth,
}

pub fn render(segments: &[&str], width: usize, sep: &str) -> BreadcrumbVerdict {
    if segments.is_empty() {
        return BreadcrumbVerdict::EmptyPath;
    }
    if width == 0 {
        return BreadcrumbVerdict::InvalidWidth;
    }
    let mut start = 0usize;
    loop {
        let visible: Vec<&&str> = segments.iter().skip(start).collect();
        let core: String = visible.iter().map(|s| **s).collect::<Vec<&str>>().join(sep);
        let candidate = if start > 0 {
            format!("…{sep}{core}")
        } else {
            core
        };
        if candidate.chars().count() <= width || start >= segments.len() - 1 {
            return BreadcrumbVerdict::Ok {
                rendered: candidate,
                hidden: start as u32,
            };
        }
        start += 1;
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_breadcrumb_path")?;

    let segs = ["root", "src", "examples", "tui", "tui_breadcrumb_path.rs"];
    println!("full: {:?}", render(&segs, 80, "/"));
    println!("narrow: {:?}", render(&segs, 20, "/"));
    println!("very narrow: {:?}", render(&segs, 5, "/"));
    println!("empty: {:?}", render(&[], 80, "/"));
    println!("invalid: {:?}", render(&["a"], 0, "/"));
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
    fn full_fits_no_hide() {
        let v = render(&["a", "b", "c"], 80, "/");
        if let BreadcrumbVerdict::Ok { rendered, hidden } = v {
            assert_eq!(rendered, "a/b/c");
            assert_eq!(hidden, 0);
        }
    }

    #[test]
    fn narrow_hides_front() {
        let segs = ["root", "src", "examples", "deeply", "nested", "here.rs"];
        let v = render(&segs, 20, "/");
        if let BreadcrumbVerdict::Ok { hidden, .. } = v {
            assert!(hidden > 0);
        }
    }

    #[test]
    fn very_narrow_keeps_last_segment() {
        let v = render(&["root", "deeply", "very", "long", "tail"], 5, "/");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("tail"));
        }
    }

    #[test]
    fn empty_path_rejected() {
        assert_eq!(render(&[], 80, "/"), BreadcrumbVerdict::EmptyPath);
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(render(&["a"], 0, "/"), BreadcrumbVerdict::InvalidWidth);
    }

    #[test]
    fn single_segment_no_hide() {
        let v = render(&["root"], 80, "/");
        if let BreadcrumbVerdict::Ok { rendered, hidden } = v {
            assert_eq!(rendered, "root");
            assert_eq!(hidden, 0);
        }
    }

    #[test]
    fn separator_used() {
        let v = render(&["a", "b"], 80, " > ");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains(" > "));
        }
    }

    #[test]
    fn ellipsis_present_when_hidden() {
        let segs = [
            "extremely",
            "long",
            "first",
            "segment",
            "names",
            "to",
            "force",
            "hide",
            "tail",
        ];
        let v = render(&segs, 15, "/");
        if let BreadcrumbVerdict::Ok { rendered, hidden } = v {
            if hidden > 0 {
                assert!(rendered.contains('…'));
            }
        }
    }

    #[test]
    fn unicode_segments_work() {
        let v = render(&["héllo", "wörld"], 80, "/");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("héllo"));
            assert!(rendered.contains("wörld"));
        }
    }

    #[test]
    fn deterministic() {
        let segs = ["a", "b", "c"];
        let a = render(&segs, 80, "/");
        let b = render(&segs, 80, "/");
        assert_eq!(a, b);
    }
}
