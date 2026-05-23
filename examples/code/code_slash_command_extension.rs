//! # apr code — Slash Command Extension
//!
//! `apr code` ships with 21 built-in slash commands (per `apr-code-parity-v1.yaml`
//! row `slash-commands` v4.2 SHIPPED). Project-local extensions live under
//! `.apr/commands/<name>.md` (or `.claude/commands/<name>.md`); each markdown
//! file declares a custom slash command's prompt template.
//!
//! This recipe writes two sample slash commands, runs the discovery routine,
//! and asserts the file naming convention (`/<stem>` becomes the slash
//! invocation name).
//!
//! Demonstrates the **C.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-SLASH-PARITY-001 (SHIPPED v4.2)
//!
//! Run with: cargo run --example code_slash_command_extension
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn discover_commands(commands_dir: &Path) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let Ok(entries) = fs::read_dir(commands_dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                found.insert(format!("/{stem}"));
            }
        }
    }
    found
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_slash_command_extension")?;
    let dir = tempfile::tempdir()?;
    let commands_dir = dir.path().join(".apr").join("commands");
    fs::create_dir_all(&commands_dir)?;

    fs::write(
        commands_dir.join("review-pr.md"),
        "Review the staged changes for clippy violations and IIUR contract drift.\n\
         Cite each finding with file:line.\n",
    )?;
    fs::write(
        commands_dir.join("explain-falsification.md"),
        "Walk the falsification gates in tests/falsification.rs.\n\
         For each F-claim, name the threshold and the test that enforces it.\n",
    )?;

    let commands = discover_commands(&commands_dir);
    println!(
        ".apr/commands discovered {} project-local slash commands:",
        commands.len()
    );
    for cmd in &commands {
        println!("  {cmd}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_discovery_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_commands_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cmds = discover_commands(&dir.path().join(".apr").join("commands"));
        assert!(cmds.is_empty());
    }

    #[test]
    fn slash_invocation_name_is_stem_prefixed() {
        let dir = tempfile::tempdir().unwrap();
        let cdir = dir.path().join(".apr").join("commands");
        fs::create_dir_all(&cdir).unwrap();
        fs::write(cdir.join("hello-world.md"), "say hi").unwrap();
        let cmds = discover_commands(&cdir);
        assert!(cmds.contains("/hello-world"));
    }

    #[test]
    fn non_md_files_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let cdir = dir.path().join(".apr").join("commands");
        fs::create_dir_all(&cdir).unwrap();
        fs::write(cdir.join("ok.md"), "ok").unwrap();
        fs::write(cdir.join("README.txt"), "skip").unwrap();
        let cmds = discover_commands(&cdir);
        assert_eq!(cmds.len(), 1);
        assert!(cmds.contains("/ok"));
    }
}
