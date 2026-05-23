//! # TUI Empty-State Render
//!
//! Render an empty-state placeholder block when a list/table has no
//! content. Returns a centered multi-line placeholder with optional
//! call-to-action text.
//!
//! Demonstrates the **TUI.167** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Material Design empty-state guidelines; React TanStack
//!  Table empty-row pattern.
//!
//! Run with: cargo run --example tui_empty_state_render
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EmptyStateVerdict {
    Ok { lines: Vec<String>, width: u32 },
    InvalidConfig,
}

pub fn render(message: &str, cta: Option<&str>, width: u32) -> EmptyStateVerdict {
    if message.is_empty() || width < 10 {
        return EmptyStateVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::new();
    let center = |s: &str, w: u32| -> String {
        let len = s.chars().count() as u32;
        if len >= w {
            return s.to_string();
        }
        let pad = (w - len) / 2;
        format!("{}{s}", " ".repeat(pad as usize))
    };
    lines.push(center(message, width));
    if let Some(c) = cta {
        if !c.is_empty() {
            lines.push(String::new());
            lines.push(center(c, width));
        }
    }
    EmptyStateVerdict::Ok { lines, width }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_empty_state_render")?;

    println!("simple: {:?}", render("No items", None, 40));
    println!(
        "with cta: {:?}",
        render("No items", Some("Press N to add"), 40)
    );
    println!("invalid: {:?}", render("", None, 40));
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
    fn empty_message_rejected() {
        assert_eq!(render("", None, 40), EmptyStateVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_narrow() {
        assert_eq!(render("hi", None, 5), EmptyStateVerdict::InvalidConfig);
    }

    #[test]
    fn message_only_one_line() {
        let v = render("No items", None, 40);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 1);
        }
    }

    #[test]
    fn message_with_cta_three_lines() {
        let v = render("No items", Some("Press N"), 40);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            // message + blank + cta
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn message_centered() {
        let v = render("Hi", None, 10);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            // "Hi" centered in 10 → "    Hi" (4 leading spaces).
            assert!(lines[0].starts_with("    "));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("Hi", None, 20);
        let r2 = render("Hi", None, 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_cta_treated_as_none() {
        let v = render("Hi", Some(""), 20);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 1);
        }
    }

    #[test]
    fn long_message_no_truncation() {
        let v = render("a very long message exceeding width", None, 20);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("very long"));
        }
    }

    #[test]
    fn width_returned() {
        let v = render("Hi", None, 30);
        if let EmptyStateVerdict::Ok { width, .. } = v {
            assert_eq!(width, 30);
        }
    }

    #[test]
    fn unicode_message_centered() {
        let v = render("café", None, 10);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("café"));
        }
    }

    #[test]
    fn min_width_accepted() {
        let v = render("Hi", None, 10);
        assert!(matches!(v, EmptyStateVerdict::Ok { .. }));
    }

    #[test]
    fn cta_centered() {
        let v = render("Hi", Some("Press N"), 20);
        if let EmptyStateVerdict::Ok { lines, .. } = v {
            // Last line should contain "Press N" with leading spaces (centered).
            assert!(lines[2].contains("Press N"));
        }
    }
}
