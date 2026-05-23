//! # apr code — Custom Agent Definition (.apr/agents/*.md)
//!
//! Demonstrates the file layout and frontmatter format an `apr code` install
//! discovers under `.apr/agents/` (and `.claude/agents/` for parity). The
//! recipe writes a minimal agent definition to a tempdir, parses the
//! `---`-fenced YAML frontmatter using the same parser shape `apr code`
//! uses (`fn parse_agent_md`, per `apr-code-parity-v1.yaml` row
//! `custom-agents` evidence_symbols).
//!
//! Demonstrates the **C.5** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-CUSTOM-AGENTS-001 (SHIPPED v4.5)
//!
//! Run with: cargo run --example code_custom_agent_definition
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::fs;
use std::io::Write;

const SAMPLE_AGENT: &str = "\
---
name: code-reviewer
description: Reviews staged Rust changes against project clippy + style rules
allowed-tools: [grep, read, edit]
model: opus-4-7
---

You are a Rust code reviewer for the apr-cookbook project. Your job is to
review staged changes and surface clippy violations, IIUR-contract
violations, and missing test coverage. Cite the exact file:line.
";

/// Hand-rolled frontmatter parser mirroring `apr code`'s `parse_agent_md`.
/// Returns (frontmatter_yaml, body) or an error if the `---` fences are missing.
fn parse_agent_md(content: &str) -> Result<(String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.first().map(|l| l.trim()) != Some("---") {
        return Err(apr_cookbook::CookbookError::Validation(
            "agent file must begin with --- frontmatter fence".into(),
        ));
    }
    let close = lines
        .iter()
        .skip(1)
        .position(|l| l.trim() == "---")
        .ok_or_else(|| {
            apr_cookbook::CookbookError::Validation(
                "agent frontmatter missing closing --- fence".into(),
            )
        })?;
    let fm = lines[1..=close].join("\n");
    let body = lines[close + 2..].join("\n");
    Ok((fm, body))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_custom_agent_definition")?;
    let dir = tempfile::tempdir()?;
    let agents_dir = dir.path().join(".apr").join("agents");
    fs::create_dir_all(&agents_dir)?;
    let agent_path = agents_dir.join("code-reviewer.md");
    let mut file = fs::File::create(&agent_path)?;
    file.write_all(SAMPLE_AGENT.as_bytes())?;
    drop(file);

    let content = fs::read_to_string(&agent_path)?;
    let (fm, body) = parse_agent_md(&content)?;
    println!(
        "discovered agent at {}",
        agent_path.strip_prefix(dir.path()).unwrap().display()
    );
    println!("frontmatter ({} bytes):", fm.len());
    println!("{fm}");
    println!(
        "body ({} bytes, first line): {}",
        body.len(),
        body.lines().next().unwrap_or("")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_def_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parser_rejects_missing_open_fence() {
        let no_open = "name: foo\n---\nbody\n";
        assert!(parse_agent_md(no_open).is_err());
    }

    #[test]
    fn parser_rejects_missing_close_fence() {
        let no_close = "---\nname: foo\nbody\n";
        assert!(parse_agent_md(no_close).is_err());
    }

    #[test]
    fn parser_extracts_frontmatter_and_body() {
        let (fm, body) = parse_agent_md(SAMPLE_AGENT).unwrap();
        assert!(fm.contains("name: code-reviewer"));
        assert!(body.contains("Rust code reviewer"));
    }
}
