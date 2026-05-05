//! # apr code — Skill Discovery (.apr/skills/*.md)
//!
//! Skills are markdown files under `.apr/skills/` (or `.claude/skills/` for
//! parity) that define reusable instructions an `apr code` session can
//! invoke by name. The discovery layout is symmetric to custom agents:
//! flat `skills/<name>.md` or sub-directory `skills/<name>/SKILL.md`.
//!
//! This recipe writes a sample skill, runs the discovery routine across
//! both layout shapes, asserts both are found, and that `.apr/skills`
//! takes precedence over `.claude/skills` on name collision.
//!
//! Demonstrates the **C.6** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-SKILLS-001 (SHIPPED v4.7)
//!
//! Run with: cargo run --example code_skill_discovery
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Discover skills under one root, supporting both flat and subdir layouts.
/// Returns a map from skill name -> absolute path.
fn discover_skills(root: &Path) -> BTreeMap<String, PathBuf> {
    let mut found = BTreeMap::new();
    let Ok(entries) = fs::read_dir(root) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                found.insert(stem.to_string(), path);
            }
        } else if path.is_dir() {
            let nested = path.join("SKILL.md");
            if nested.exists() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    found.insert(name.to_string(), nested);
                }
            }
        }
    }
    found
}

/// Merge .apr-discovered + .claude-discovered with .apr taking precedence.
fn merge_with_apr_precedence(
    apr: BTreeMap<String, PathBuf>,
    claude: BTreeMap<String, PathBuf>,
) -> BTreeMap<String, PathBuf> {
    let mut out = claude;
    out.extend(apr);
    out
}

fn write_skill(path: &Path, name: &str, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = format!("---\nname: {name}\n---\n\n{body}\n");
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_skill_discovery")?;
    let dir = tempfile::tempdir()?;
    let apr_skills = dir.path().join(".apr").join("skills");
    let claude_skills = dir.path().join(".claude").join("skills");

    write_skill(
        &apr_skills.join("flat-skill.md"),
        "flat-skill",
        "flat layout",
    )?;
    write_skill(
        &apr_skills.join("nested-skill").join("SKILL.md"),
        "nested-skill",
        "nested subdir layout",
    )?;
    write_skill(
        &claude_skills.join("flat-skill.md"),
        "flat-skill",
        "claude version (overridden by .apr)",
    )?;
    write_skill(
        &claude_skills.join("claude-only-skill.md"),
        "claude-only-skill",
        "claude-exclusive",
    )?;

    let apr_discovered = discover_skills(&apr_skills);
    let claude_discovered = discover_skills(&claude_skills);
    let merged = merge_with_apr_precedence(apr_discovered.clone(), claude_discovered);

    println!(
        ".apr/skills found: {} ({})",
        apr_discovered.len(),
        apr_discovered
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("merged skills: {}", merged.len());
    for (name, path) in &merged {
        let rel = path.strip_prefix(dir.path()).unwrap();
        println!("  {name} -> {}", rel.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn flat_and_nested_layouts_both_discovered() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("skills");
        write_skill(&root.join("a.md"), "a", "flat").unwrap();
        write_skill(&root.join("b").join("SKILL.md"), "b", "nested").unwrap();
        let found = discover_skills(&root);
        assert!(found.contains_key("a"));
        assert!(found.contains_key("b"));
    }

    #[test]
    fn apr_precedence_over_claude_on_name_collision() {
        let mut apr = BTreeMap::new();
        apr.insert("foo".into(), PathBuf::from("/apr/foo.md"));
        let mut claude = BTreeMap::new();
        claude.insert("foo".into(), PathBuf::from("/claude/foo.md"));
        claude.insert("only-claude".into(), PathBuf::from("/claude/only.md"));
        let merged = merge_with_apr_precedence(apr, claude);
        assert_eq!(merged["foo"], PathBuf::from("/apr/foo.md"));
        assert_eq!(merged["only-claude"], PathBuf::from("/claude/only.md"));
    }
}
