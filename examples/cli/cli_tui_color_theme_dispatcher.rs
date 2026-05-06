//! # apr tui --theme — Color Theme Dispatcher
//!
//! TUI themes: dark/light/auto. `auto` reads the COLORFGBG env var
//! (X11/iTerm convention) — value `15;0` = light text on dark bg →
//! Dark, `0;15` = dark text on light bg → Light. Color depth probed
//! via NO_COLOR (off) / COLORTERM (truecolor) / TERM (256/16). This
//! recipe builds the dispatcher.
//!
//! Demonstrates the **TUI.4** recipe for PMAT-122 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-001 + no-color.org + colorterm conventions
//!
//! Run with: cargo run --example cli_tui_color_theme_dispatcher
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Theme {
    Dark,
    Light,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ColorDepth {
    None,
    Sixteen,
    TwoFiftySix,
    Truecolor,
}

pub fn detect_theme(explicit: Option<&str>, color_fg_bg: Option<&str>) -> Theme {
    if let Some(s) = explicit {
        match s {
            "dark" => return Theme::Dark,
            "light" => return Theme::Light,
            _ => {}
        }
    }
    if let Some(s) = color_fg_bg {
        // "<fg>;<bg>" — bg < 8 → dark; bg >= 8 → light.
        if let Some((_, bg)) = s.split_once(';') {
            if let Ok(bg_idx) = bg.parse::<u8>() {
                return if bg_idx < 8 {
                    Theme::Dark
                } else {
                    Theme::Light
                };
            }
        }
    }
    Theme::Dark
}

pub fn detect_depth(
    no_color: Option<&str>,
    colorterm: Option<&str>,
    term: Option<&str>,
) -> ColorDepth {
    if no_color.is_some_and(|v| !v.is_empty()) {
        return ColorDepth::None;
    }
    if let Some(c) = colorterm {
        if c.eq_ignore_ascii_case("truecolor") || c.eq_ignore_ascii_case("24bit") {
            return ColorDepth::Truecolor;
        }
    }
    if let Some(t) = term {
        if t.contains("256") {
            return ColorDepth::TwoFiftySix;
        }
        if t.contains("color") {
            return ColorDepth::Sixteen;
        }
    }
    ColorDepth::None
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_color_theme_dispatcher")?;

    let cases = [
        (None, None),
        (Some("dark"), None),
        (Some("light"), None),
        (None, Some("15;0")),
        (None, Some("0;15")),
    ];
    for (e, fb) in cases {
        println!("explicit={e:?} fgbg={fb:?}  →  {:?}", detect_theme(e, fb));
    }

    let depth_cases = [
        (None, None, Some("xterm")),
        (Some("1"), None, Some("xterm-256color")),
        (None, Some("truecolor"), Some("xterm-256color")),
        (None, None, Some("xterm-256color")),
    ];
    for (nc, ct, t) in depth_cases {
        println!(
            "no_color={nc:?} colorterm={ct:?} term={t:?}  →  {:?}",
            detect_depth(nc, ct, t)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn explicit_dark_returns_dark() {
        assert_eq!(detect_theme(Some("dark"), None), Theme::Dark);
    }

    #[test]
    fn explicit_light_returns_light() {
        assert_eq!(detect_theme(Some("light"), None), Theme::Light);
    }

    #[test]
    fn explicit_overrides_fgbg() {
        // explicit "dark" wins over fgbg suggesting light.
        assert_eq!(detect_theme(Some("dark"), Some("0;15")), Theme::Dark);
    }

    #[test]
    fn fgbg_dark_bg_gives_dark_theme() {
        // bg=0 is black → dark theme.
        assert_eq!(detect_theme(None, Some("15;0")), Theme::Dark);
    }

    #[test]
    fn fgbg_light_bg_gives_light_theme() {
        // bg=15 is white → light theme.
        assert_eq!(detect_theme(None, Some("0;15")), Theme::Light);
    }

    #[test]
    fn no_signals_defaults_to_dark() {
        assert_eq!(detect_theme(None, None), Theme::Dark);
    }

    #[test]
    fn no_color_set_disables_color() {
        assert_eq!(
            detect_depth(Some("1"), Some("truecolor"), Some("xterm-256color")),
            ColorDepth::None
        );
    }

    #[test]
    fn empty_no_color_does_not_disable() {
        // NO_COLOR semantics: presence is what counts; empty is treated as unset.
        assert_eq!(
            detect_depth(Some(""), None, Some("xterm-256color")),
            ColorDepth::TwoFiftySix
        );
    }

    #[test]
    fn colorterm_truecolor_wins() {
        assert_eq!(
            detect_depth(None, Some("truecolor"), Some("xterm")),
            ColorDepth::Truecolor
        );
        assert_eq!(
            detect_depth(None, Some("24BIT"), None),
            ColorDepth::Truecolor
        );
    }

    #[test]
    fn term_256color_detected() {
        assert_eq!(
            detect_depth(None, None, Some("xterm-256color")),
            ColorDepth::TwoFiftySix
        );
    }

    #[test]
    fn term_color_only_detected() {
        assert_eq!(
            detect_depth(None, None, Some("xterm-color")),
            ColorDepth::Sixteen
        );
    }
}
