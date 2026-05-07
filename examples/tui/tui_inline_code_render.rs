//! # TUI Inline Code Render
//!
//! Render text fragments alternating between prose and inline code,
//! wrapping code spans with backticks. Returns rendered string.
//!
//! Demonstrates the **TUI.124** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CommonMark §6.1 (Code spans); GFM inline code rendering.
//!
//! Run with: cargo run --example tui_inline_code_render
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FragmentKind {
    Prose,
    Code,
}

#[derive(Debug, PartialEq)]
pub enum CodeRenderVerdict {
    Ok { rendered: String, code_count: u32 },
    InvalidConfig,
}

pub fn render(fragments: &[(FragmentKind, &str)]) -> CodeRenderVerdict {
    if fragments.is_empty() {
        return CodeRenderVerdict::InvalidConfig;
    }
    let mut rendered = String::new();
    let mut code_count = 0u32;
    for (kind, text) in fragments {
        match kind {
            FragmentKind::Prose => rendered.push_str(text),
            FragmentKind::Code => {
                code_count += 1;
                rendered.push('`');
                rendered.push_str(text);
                rendered.push('`');
            }
        }
    }
    CodeRenderVerdict::Ok {
        rendered,
        code_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_inline_code_render")?;

    let fragments = [
        (FragmentKind::Prose, "Use "),
        (FragmentKind::Code, "cargo build"),
        (FragmentKind::Prose, " to compile."),
    ];
    println!("rendered: {:?}", render(&fragments));
    println!("invalid: {:?}", render(&[]));
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
    fn code_wrapped_with_backticks() {
        let fragments = [(FragmentKind::Code, "ls")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "`ls`");
        }
    }

    #[test]
    fn prose_unwrapped() {
        let fragments = [(FragmentKind::Prose, "hello")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "hello");
        }
    }

    #[test]
    fn mixed_renders_correctly() {
        let fragments = [
            (FragmentKind::Prose, "Run "),
            (FragmentKind::Code, "ls -l"),
            (FragmentKind::Prose, " now"),
        ];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "Run `ls -l` now");
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(render(&[]), CodeRenderVerdict::InvalidConfig);
    }

    #[test]
    fn code_count_correct() {
        let fragments = [
            (FragmentKind::Code, "a"),
            (FragmentKind::Prose, "x"),
            (FragmentKind::Code, "b"),
        ];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { code_count, .. } = v {
            assert_eq!(code_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let fragments = [(FragmentKind::Code, "x")];
        let r1 = render(&fragments);
        let r2 = render(&fragments);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_code_supported() {
        let fragments = [(FragmentKind::Code, "café")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "`café`");
        }
    }

    #[test]
    fn empty_fragment_text_works() {
        let fragments = [(FragmentKind::Code, "")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "``");
        }
    }

    #[test]
    fn many_fragments_handled() {
        let fragments: Vec<(FragmentKind, &str)> =
            (0..20).map(|_| (FragmentKind::Prose, "x")).collect();
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.len(), 20);
        }
    }

    #[test]
    fn no_code_count_zero() {
        let fragments = [(FragmentKind::Prose, "all prose")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { code_count, .. } = v {
            assert_eq!(code_count, 0);
        }
    }

    #[test]
    fn all_code_count_correct() {
        let fragments = [(FragmentKind::Code, "a"), (FragmentKind::Code, "b")];
        let v = render(&fragments);
        if let CodeRenderVerdict::Ok { code_count, .. } = v {
            assert_eq!(code_count, 2);
        }
    }
}
