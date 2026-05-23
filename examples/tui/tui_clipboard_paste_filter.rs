//! # TUI Clipboard Paste Filter
//!
//! Sanitize a paste-buffer string before insertion: strip control
//! chars, normalize CRLF→LF, optionally trim trailing whitespace.
//! Returns sanitized string and removed-char count.
//!
//! Demonstrates the **TUI.83** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim 'pastetoggle' / nvim ":set paste"; xterm bracketed
//!  paste mode (CSI 2004).
//!
//! Run with: cargo run --example tui_clipboard_paste_filter
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PasteVerdict {
    Ok {
        sanitized: String,
        removed_chars: u32,
    },
    InvalidConfig,
}

pub fn sanitize(text: &str, trim_trailing_ws: bool) -> PasteVerdict {
    if text.is_empty() {
        return PasteVerdict::InvalidConfig;
    }
    let normalized = text.replace("\r\n", "\n");
    let mut sanitized = String::with_capacity(normalized.len());
    let mut removed: u32 = 0;
    for c in normalized.chars() {
        if c == '\n' || c == '\t' {
            sanitized.push(c);
        } else if c.is_control() || c == '\x7f' {
            removed += 1;
        } else {
            sanitized.push(c);
        }
    }
    if trim_trailing_ws {
        let trimmed: String = sanitized
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        let trimmed_count = (sanitized.chars().count() - trimmed.chars().count()) as u32;
        sanitized = trimmed;
        removed += trimmed_count;
    }
    let crlf_removed = (text.chars().count() - normalized.chars().count()) as u32;
    removed += crlf_removed;
    PasteVerdict::Ok {
        sanitized,
        removed_chars: removed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_clipboard_paste_filter")?;

    println!("plain: {:?}", sanitize("hello world", false));
    println!("crlf: {:?}", sanitize("a\r\nb", false));
    println!("control: {:?}", sanitize("hi\x07there", false));
    println!("trim: {:?}", sanitize("a   \nb", true));
    println!("invalid: {:?}", sanitize("", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn plain_text_unchanged() {
        let v = sanitize("hello world", false);
        if let PasteVerdict::Ok {
            sanitized,
            removed_chars,
        } = v
        {
            assert_eq!(sanitized, "hello world");
            assert_eq!(removed_chars, 0);
        }
    }

    #[test]
    fn crlf_normalized() {
        let v = sanitize("a\r\nb", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "a\nb");
        }
    }

    #[test]
    fn control_char_removed() {
        let v = sanitize("hi\x07there", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "hithere");
        }
    }

    #[test]
    fn newline_preserved() {
        let v = sanitize("line1\nline2", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert!(sanitized.contains('\n'));
        }
    }

    #[test]
    fn tab_preserved() {
        let v = sanitize("a\tb", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert!(sanitized.contains('\t'));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(sanitize("", false), PasteVerdict::InvalidConfig);
    }

    #[test]
    fn trailing_whitespace_trimmed() {
        let v = sanitize("a   \nb", true);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "a\nb");
        }
    }

    #[test]
    fn trailing_whitespace_kept_without_flag() {
        let v = sanitize("a   ", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "a   ");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = sanitize("hello", false);
        let r2 = sanitize("hello", false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn delete_char_removed() {
        let v = sanitize("a\x7fb", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "ab");
        }
    }

    #[test]
    fn removed_count_accurate() {
        let v = sanitize("hi\x07\x08there", false);
        if let PasteVerdict::Ok { removed_chars, .. } = v {
            assert_eq!(removed_chars, 2);
        }
    }

    #[test]
    fn unicode_text_preserved() {
        let v = sanitize("café", false);
        if let PasteVerdict::Ok { sanitized, .. } = v {
            assert_eq!(sanitized, "café");
        }
    }
}
