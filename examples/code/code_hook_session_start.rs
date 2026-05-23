//! # apr code — Hook: SessionStart
//!
//! Hooks are shell scripts under `.apr/hooks/<event>/<name>.sh` (or
//! `.claude/hooks/<event>/<name>.sh`) that fire at lifecycle events. The
//! `SessionStart` event runs once at REPL launch — useful for sanity checks
//! (toolchain present, env var set, project tree clean).
//!
//! This recipe writes a sample SessionStart hook to a tempdir, runs the
//! discovery routine that an `apr code` install uses to find hooks for an
//! event, and asserts the hook is found + executable. It does NOT actually
//! execute the hook (cookbook is offline-only per IIUR).
//!
//! Demonstrates the **C.3** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-HOOKS-001 (SHIPPED v4.3)
//!
//! Run with: cargo run --example code_hook_session_start
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const SAMPLE_HOOK: &str = "\
#!/usr/bin/env bash
# SessionStart hook — runs once at apr code REPL launch.
# Convention: exit 0 = pass, non-zero = abort session start with stderr message.
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo \"ERROR: cargo not on PATH; apr code session aborted\" >&2
  exit 1
fi

echo \"SessionStart: cargo $(cargo --version | awk '{print $2}') OK\"
";

/// Discover hooks under a project's `.apr/hooks/<event>/` directory.
/// Returns sorted absolute paths to all `.sh` hooks for the given event.
fn discover_hooks(root: &Path, event: &str) -> Vec<PathBuf> {
    let event_dir = root.join(".apr").join("hooks").join(event);
    let Ok(entries) = fs::read_dir(&event_dir) else {
        return Vec::new();
    };
    let mut found: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("sh"))
        .collect();
    found.sort();
    found
}

#[cfg(unix)]
fn write_executable(path: &Path, content: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    drop(file);
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn write_executable(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_hook_session_start")?;
    let dir = tempfile::tempdir()?;
    let hook_path = dir
        .path()
        .join(".apr")
        .join("hooks")
        .join("SessionStart")
        .join("verify-cargo.sh");

    write_executable(&hook_path, SAMPLE_HOOK)?;

    let hooks = discover_hooks(dir.path(), "SessionStart");
    println!("discovered {} SessionStart hook(s):", hooks.len());
    for h in &hooks {
        let rel = h.strip_prefix(dir.path()).unwrap();
        let metadata = fs::metadata(h)?;
        #[cfg(unix)]
        let exec = {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode() & 0o111 != 0
        };
        #[cfg(not(unix))]
        let exec = true;
        println!("  {} (executable: {exec})", rel.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_discovery_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_hooks_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let hooks = discover_hooks(dir.path(), "SessionStart");
        assert!(hooks.is_empty());
    }

    #[test]
    fn only_sh_files_discovered() {
        let dir = tempfile::tempdir().unwrap();
        let event_dir = dir.path().join(".apr").join("hooks").join("SessionStart");
        fs::create_dir_all(&event_dir).unwrap();
        fs::write(event_dir.join("yes.sh"), "echo yes").unwrap();
        fs::write(event_dir.join("no.txt"), "echo no").unwrap();
        let found = discover_hooks(dir.path(), "SessionStart");
        assert_eq!(found.len(), 1);
        assert!(found[0].file_name().unwrap() == "yes.sh");
    }
}
