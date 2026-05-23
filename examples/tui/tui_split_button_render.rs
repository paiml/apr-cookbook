//! # TUI Split Button Render
//!
//! Render a split-button widget like `[ Save | ▾ ]` that has a primary
//! action label plus a dropdown trigger. Returns rendered string and
//! the click-zone boundaries.
//!
//! Demonstrates the **TUI.79** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Cocoa NSPopUpButton; Material Design split button
//!  guidelines.
//!
//! Run with: cargo run --example tui_split_button_render
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ButtonVerdict {
    Ok {
        rendered: String,
        primary_zone: (u32, u32),
        dropdown_zone: (u32, u32),
    },
    InvalidConfig,
}

pub fn render(label: &str) -> ButtonVerdict {
    if label.is_empty() {
        return ButtonVerdict::InvalidConfig;
    }
    let primary = format!("[ {label} | ▾ ]");
    let label_chars = label.chars().count() as u32;
    // Layout: "[ " (2) + label (label_chars) + " " (1) = primary end.
    // Then "| ▾ ]" starts.
    let primary_start = 0u32;
    let primary_end = 2 + label_chars + 1; // exclusive
    let dropdown_start = primary_end + 1; // skip "|"
    let dropdown_end = dropdown_start + 1;
    ButtonVerdict::Ok {
        rendered: primary,
        primary_zone: (primary_start, primary_end),
        dropdown_zone: (dropdown_start, dropdown_end),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_split_button_render")?;

    println!("save: {:?}", render("Save"));
    println!("commit: {:?}", render("Commit and push"));
    println!("invalid: {:?}", render(""));
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
    fn save_button_correct() {
        let v = render("Save");
        if let ButtonVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "[ Save | ▾ ]");
        }
    }

    #[test]
    fn empty_label_rejected() {
        assert_eq!(render(""), ButtonVerdict::InvalidConfig);
    }

    #[test]
    fn unicode_label_supported() {
        let v = render("résumé");
        if let ButtonVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("résumé"));
            assert!(rendered.contains('▾'));
        }
    }

    #[test]
    fn dropdown_marker_present() {
        let v = render("X");
        if let ButtonVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('▾'));
        }
    }

    #[test]
    fn brackets_around_widget() {
        let v = render("X");
        if let ButtonVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('['));
            assert!(rendered.ends_with(']'));
        }
    }

    #[test]
    fn primary_zone_starts_at_zero() {
        let v = render("Save");
        if let ButtonVerdict::Ok {
            primary_zone: (s, _),
            ..
        } = v
        {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn primary_zone_excludes_separator() {
        let v = render("Save");
        if let ButtonVerdict::Ok {
            primary_zone: (_, e),
            dropdown_zone: (s, _),
            ..
        } = v
        {
            assert!(s > e);
        }
    }

    #[test]
    fn dropdown_zone_smaller_than_primary() {
        let v = render("Long Label");
        if let ButtonVerdict::Ok {
            primary_zone,
            dropdown_zone,
            ..
        } = v
        {
            let p_size = primary_zone.1 - primary_zone.0;
            let d_size = dropdown_zone.1 - dropdown_zone.0;
            assert!(p_size > d_size);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("Save");
        let r2 = render("Save");
        assert_eq!(r1, r2);
    }

    #[test]
    fn label_with_spaces() {
        let v = render("OK and apply");
        if let ButtonVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("OK and apply"));
        }
    }

    #[test]
    fn dropdown_zone_one_char_wide() {
        let v = render("Save");
        if let ButtonVerdict::Ok {
            dropdown_zone: (s, e),
            ..
        } = v
        {
            assert_eq!(e - s, 1);
        }
    }
}
