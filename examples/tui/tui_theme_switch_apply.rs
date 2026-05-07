//! # TUI Theme Switch Apply
//!
//! Apply a theme switch with valid transitions. Allowed: any → any,
//! but rejects identical (no-op) and unknown theme names.
//! Returns the new theme + a transition counter.
//!
//! Demonstrates the **TUI.135** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `colorscheme` command; bat/dotfiles theme-switching
//!  patterns.
//!
//! Run with: cargo run --example tui_theme_switch_apply
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThemeVerdict {
    Switched { new_theme: String, count: u32 },
    Unchanged,
    UnknownTheme,
    InvalidConfig,
}

pub fn switch(
    available: &[&str],
    current: &str,
    requested: &str,
    prior_count: u32,
) -> ThemeVerdict {
    if available.is_empty() {
        return ThemeVerdict::InvalidConfig;
    }
    if !available.contains(&requested) {
        return ThemeVerdict::UnknownTheme;
    }
    if current == requested {
        return ThemeVerdict::Unchanged;
    }
    ThemeVerdict::Switched {
        new_theme: requested.to_string(),
        count: prior_count + 1,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_theme_switch_apply")?;

    let themes = ["dark", "light", "solarized"];
    println!("switch: {:?}", switch(&themes, "dark", "light", 5));
    println!("noop: {:?}", switch(&themes, "dark", "dark", 5));
    println!("unknown: {:?}", switch(&themes, "dark", "neon", 5));
    println!("invalid: {:?}", switch(&[], "dark", "light", 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_switch_increments_count() {
        let v = switch(&["a", "b"], "a", "b", 5);
        assert_eq!(
            v,
            ThemeVerdict::Switched {
                new_theme: "b".to_string(),
                count: 6,
            }
        );
    }

    #[test]
    fn same_theme_unchanged() {
        let v = switch(&["a", "b"], "a", "a", 5);
        assert_eq!(v, ThemeVerdict::Unchanged);
    }

    #[test]
    fn unknown_theme_rejected() {
        let v = switch(&["a", "b"], "a", "c", 5);
        assert_eq!(v, ThemeVerdict::UnknownTheme);
    }

    #[test]
    fn empty_available_rejected() {
        let v = switch(&[], "a", "b", 5);
        assert_eq!(v, ThemeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = switch(&["a", "b"], "a", "b", 5);
        let r2 = switch(&["a", "b"], "a", "b", 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_starts_at_zero() {
        let v = switch(&["a", "b"], "a", "b", 0);
        if let ThemeVerdict::Switched { count, .. } = v {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn case_sensitive_theme() {
        let v = switch(&["Dark"], "Dark", "dark", 5);
        assert_eq!(v, ThemeVerdict::UnknownTheme);
    }

    #[test]
    fn unknown_current_with_valid_request_switches() {
        // current="" not in available; requested is. We accept switch.
        let v = switch(&["a", "b"], "", "a", 5);
        if let ThemeVerdict::Switched { new_theme, .. } = v {
            assert_eq!(new_theme, "a");
        }
    }

    #[test]
    fn many_themes_supported() {
        let themes: Vec<&str> = vec!["a", "b", "c", "d", "e", "f", "g", "h"];
        let v = switch(&themes, "a", "h", 0);
        assert!(matches!(v, ThemeVerdict::Switched { .. }));
    }

    #[test]
    fn unicode_theme_supported() {
        let v = switch(&["café"], "café", "café", 5);
        assert_eq!(v, ThemeVerdict::Unchanged);
    }

    #[test]
    fn high_count_increments() {
        let v = switch(&["a", "b"], "a", "b", 1_000_000);
        if let ThemeVerdict::Switched { count, .. } = v {
            assert_eq!(count, 1_000_001);
        }
    }
}
