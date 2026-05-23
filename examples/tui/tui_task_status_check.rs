//! # TUI Task Status Check
//!
//! Render a task with status indicator: ✓ done, ✗ failed, ● in
//! progress, ○ pending. Returns rendered string.
//!
//! Demonstrates the **TUI.174** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub Actions check-suite glyphs; ARIA `aria-busy`
//!  status conventions.
//!
//! Run with: cargo run --example tui_task_status_check
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TaskStatus {
    Done,
    Failed,
    InProgress,
    Pending,
}

#[derive(Debug, PartialEq)]
pub enum TaskRenderVerdict {
    Ok { rendered: String, glyph: char },
    InvalidConfig,
}

pub fn render(label: &str, status: &TaskStatus) -> TaskRenderVerdict {
    if label.is_empty() {
        return TaskRenderVerdict::InvalidConfig;
    }
    let glyph = match status {
        TaskStatus::Done => '✓',
        TaskStatus::Failed => '✗',
        TaskStatus::InProgress => '●',
        TaskStatus::Pending => '○',
    };
    TaskRenderVerdict::Ok {
        rendered: format!("{glyph} {label}"),
        glyph,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_task_status_check")?;

    println!("done: {:?}", render("Build", &TaskStatus::Done));
    println!("failed: {:?}", render("Test", &TaskStatus::Failed));
    println!(
        "in-progress: {:?}",
        render("Deploy", &TaskStatus::InProgress)
    );
    println!("pending: {:?}", render("Notify", &TaskStatus::Pending));
    println!("invalid: {:?}", render("", &TaskStatus::Done));
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
    fn empty_label_rejected() {
        assert_eq!(
            render("", &TaskStatus::Done),
            TaskRenderVerdict::InvalidConfig
        );
    }

    #[test]
    fn done_glyph_check() {
        let v = render("X", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { glyph, .. } = v {
            assert_eq!(glyph, '✓');
        }
    }

    #[test]
    fn failed_glyph_x() {
        let v = render("X", &TaskStatus::Failed);
        if let TaskRenderVerdict::Ok { glyph, .. } = v {
            assert_eq!(glyph, '✗');
        }
    }

    #[test]
    fn in_progress_glyph_filled_circle() {
        let v = render("X", &TaskStatus::InProgress);
        if let TaskRenderVerdict::Ok { glyph, .. } = v {
            assert_eq!(glyph, '●');
        }
    }

    #[test]
    fn pending_glyph_open_circle() {
        let v = render("X", &TaskStatus::Pending);
        if let TaskRenderVerdict::Ok { glyph, .. } = v {
            assert_eq!(glyph, '○');
        }
    }

    #[test]
    fn label_in_rendered() {
        let v = render("Build", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("Build"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("X", &TaskStatus::Done);
        let r2 = render("X", &TaskStatus::Done);
        assert_eq!(r1, r2);
    }

    #[test]
    fn glyphs_unique_per_status() {
        let d = render("X", &TaskStatus::Done);
        let f = render("X", &TaskStatus::Failed);
        let i = render("X", &TaskStatus::InProgress);
        let p = render("X", &TaskStatus::Pending);
        if let (
            TaskRenderVerdict::Ok { glyph: g_d, .. },
            TaskRenderVerdict::Ok { glyph: g_f, .. },
            TaskRenderVerdict::Ok { glyph: g_i, .. },
            TaskRenderVerdict::Ok { glyph: g_p, .. },
        ) = (d, f, i, p)
        {
            assert_ne!(g_d, g_f);
            assert_ne!(g_f, g_i);
            assert_ne!(g_i, g_p);
            assert_ne!(g_d, g_p);
        }
    }

    #[test]
    fn rendered_starts_with_glyph() {
        let v = render("Test", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { rendered, glyph } = v {
            assert!(rendered.starts_with(glyph));
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render("café", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn long_label_handled() {
        let v = render("a very long task description", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("very long"));
        }
    }

    #[test]
    fn space_separator_present() {
        let v = render("X", &TaskStatus::Done);
        if let TaskRenderVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains(' '));
        }
    }
}
