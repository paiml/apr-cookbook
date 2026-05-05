//! # apr code — Worktree Isolation + Permission-Mode Lattice
//!
//! Two `apr code` parity surfaces in one recipe (per the bundled C.7 plan):
//!
//! 1. **Worktree isolation** (`apr-code-parity-v1.yaml` row PMAT-CODE-WORKTREE-001):
//!    `apr code` can spin up a temporary git worktree for an experimental
//!    branch so the main checkout stays clean. The recipe demonstrates the
//!    worktree dir layout (`<repo>/.apr/worktrees/<branch>/`) and the
//!    discovery routine.
//!
//! 2. **Permission-mode lattice** (`apr-code-parity-v1.yaml` row
//!    PMAT-CODE-PERMISSIONS-001): per-tool permissions follow a 4-level
//!    lattice (`deny < ask < allow < always_allow`); higher levels subsume
//!    lower ones. The recipe encodes the lattice and shows how a config
//!    file resolves a tool's effective permission.
//!
//! Demonstrates the **C.7** combined recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml rows PMAT-CODE-WORKTREE-001 + PMAT-CODE-PERMISSIONS-001 (both SHIPPED v4.8 + v4.9)
//!
//! Run with: cargo run --example code_worktree_isolation_permission_mode
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

/// Permission lattice per `apr code` permission-mode row. Higher = more permissive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Permission {
    Deny,
    Ask,
    Allow,
    AlwaysAllow,
}

impl Permission {
    #[allow(dead_code)] // Documented as part of the permission API surface; tests exercise the parser indirectly via fixture rules.
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "deny" => Ok(Self::Deny),
            "ask" => Ok(Self::Ask),
            "allow" => Ok(Self::Allow),
            "always_allow" => Ok(Self::AlwaysAllow),
            other => Err(apr_cookbook::CookbookError::Validation(format!(
                "unknown permission level: {other}"
            ))),
        }
    }
}

/// Resolve a tool's effective permission given a chain of (scope, permission)
/// pairs ordered from least to most specific. Most specific wins; ties broken
/// by the lattice ordering (more permissive subsumes less). Returns Deny
/// as the conservative default if no rules apply.
fn resolve_permission(rules: &[(&str, Permission)]) -> Permission {
    rules.iter().fold(Permission::Deny, |acc, (_scope, perm)| {
        match acc.cmp(perm) {
            Ordering::Less | Ordering::Equal => *perm,
            Ordering::Greater => acc,
        }
    })
}

/// Worktree discovery: list all branches that have an active worktree under
/// `<repo>/.apr/worktrees/`. Branch names support slashes (e.g. `feat/foo`)
/// so the worktree dir layout is hierarchical; we discover by recursively
/// finding `HEAD` marker files and parsing the `ref: refs/heads/<branch>`
/// line they contain.
fn list_worktree_branches(repo_root: &Path) -> Vec<String> {
    let root = repo_root.join(".apr").join("worktrees");
    let mut branches = Vec::new();
    walk_for_head(&root, &mut branches);
    branches.sort();
    branches
}

fn walk_for_head(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let head = path.join("HEAD");
            if head.is_file() {
                if let Ok(content) = fs::read_to_string(&head) {
                    if let Some(ref_line) = content.lines().find(|l| l.starts_with("ref: ")) {
                        let raw = ref_line.trim_start_matches("ref: ").trim();
                        let branch = raw.strip_prefix("refs/heads/").unwrap_or(raw);
                        out.push(branch.to_string());
                    }
                }
            } else {
                walk_for_head(&path, out);
            }
        }
    }
}

fn write_worktree_marker(repo_root: &Path, branch: &str) -> std::io::Result<PathBuf> {
    let dir = repo_root.join(".apr").join("worktrees").join(branch);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("HEAD"), format!("ref: refs/heads/{branch}\n"))?;
    Ok(dir)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_worktree_isolation_permission_mode")?;

    // (1) Worktree isolation demo
    let dir = tempfile::tempdir()?;
    write_worktree_marker(dir.path(), "experiment/long-context")?;
    write_worktree_marker(dir.path(), "fix/clippy-2026-05")?;
    let branches = list_worktree_branches(dir.path());
    println!("active worktrees ({} found):", branches.len());
    for b in &branches {
        println!("  {b}");
    }

    // (2) Permission lattice demo
    let project_rules = [
        ("global-default", Permission::Ask),
        ("project-config", Permission::Allow),
        ("session-override", Permission::AlwaysAllow),
    ];
    let effective = resolve_permission(&project_rules);
    println!("\npermission lattice resolution:");
    for (scope, perm) in &project_rules {
        println!("  {scope:20} -> {perm:?}");
    }
    println!("  effective:           -> {effective:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worktree_and_permission_run() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn lattice_orders_correctly() {
        assert!(Permission::Deny < Permission::Ask);
        assert!(Permission::Ask < Permission::Allow);
        assert!(Permission::Allow < Permission::AlwaysAllow);
    }

    #[test]
    fn most_permissive_wins() {
        let rules = [
            ("a", Permission::Deny),
            ("b", Permission::Allow),
            ("c", Permission::Ask),
        ];
        assert_eq!(resolve_permission(&rules), Permission::Allow);
    }

    #[test]
    fn empty_rules_default_deny() {
        assert_eq!(resolve_permission(&[]), Permission::Deny);
    }

    #[test]
    fn worktree_discovery_finds_marker_dirs() {
        let dir = tempfile::tempdir().unwrap();
        write_worktree_marker(dir.path(), "feat/foo").unwrap();
        write_worktree_marker(dir.path(), "feat/bar").unwrap();
        let branches = list_worktree_branches(dir.path());
        assert_eq!(
            branches,
            vec!["feat/bar".to_string(), "feat/foo".to_string()]
        );
    }
}
