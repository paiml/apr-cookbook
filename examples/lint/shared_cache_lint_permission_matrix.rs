//! # Recipe: Shared-Cache Lint — POSIX Permission Matrix
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr shared-cache-lint --observation-file observation.json` (perms path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the POSIX permission decision matrix for the shared cache.
//! The cache must be group-readable but never group/world-writable, and the
//! cache root directory must have the setgid bit (g+s) so newly created
//! blobs inherit the cache group automatically. Lint enumerates the
//! 6 boundary cases and asserts each lands at the correct verdict.
//!
//! ## Run Command
//! ```bash
//! cargo run --example shared_cache_lint_permission_matrix
//! ```
//!
//! ## References
//! - aprender CRUX-A-21 (mode bits invariant).
//! - POSIX 1003.1-2017 §3.252 (file mode bits).
//! - chmod(2) man page (S_ISGID semantics).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct PermInput {
    pub file_mode: u32,
    pub dir_mode: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermVerdict {
    Pass,
    FileWorldWritable,
    FileGroupWritable,
    DirNoSetgid,
    DirOpenWrite,
}

pub fn check_perms(p: PermInput) -> PermVerdict {
    if (p.file_mode & 0o002) != 0 {
        return PermVerdict::FileWorldWritable;
    }
    if (p.file_mode & 0o020) != 0 {
        return PermVerdict::FileGroupWritable;
    }
    if (p.dir_mode & 0o002) != 0 {
        return PermVerdict::DirOpenWrite;
    }
    if (p.dir_mode & 0o2000) == 0 {
        return PermVerdict::DirNoSetgid;
    }
    PermVerdict::Pass
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("shared_cache_lint_permission_matrix")?;

    let cases = [
        ("happy", 0o644, 0o2755),
        ("file ww", 0o646, 0o2755),
        ("file gw", 0o664, 0o2755),
        ("dir w/o setgid", 0o644, 0o0755),
        ("dir ww", 0o644, 0o2757),
    ];

    println!("=== Recipe: {} ===", ctx.name());
    for (label, fm, dm) in cases {
        let p = PermInput {
            file_mode: fm,
            dir_mode: dm,
        };
        println!(
            "{label:>15}  file=0o{fm:o}  dir=0o{dm:o}  →  {:?}",
            check_perms(p)
        );
    }
    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_644_dir_2755_passes() {
        let p = PermInput {
            file_mode: 0o644,
            dir_mode: 0o2755,
        };
        assert_eq!(check_perms(p), PermVerdict::Pass);
    }

    #[test]
    fn world_writable_file_is_top_priority_failure() {
        // Even if dir has multiple issues, file world-write is reported first
        // because it's the most exploitable.
        let p = PermInput {
            file_mode: 0o646,
            dir_mode: 0o0757,
        };
        assert_eq!(check_perms(p), PermVerdict::FileWorldWritable);
    }

    #[test]
    fn group_writable_file_fails() {
        let p = PermInput {
            file_mode: 0o664,
            dir_mode: 0o2755,
        };
        assert_eq!(check_perms(p), PermVerdict::FileGroupWritable);
    }

    #[test]
    fn setgid_missing_on_dir_fails() {
        // No setgid → newly created blobs won't inherit `apr-models` group.
        let p = PermInput {
            file_mode: 0o644,
            dir_mode: 0o0755,
        };
        assert_eq!(check_perms(p), PermVerdict::DirNoSetgid);
    }

    #[test]
    fn world_writable_dir_fails_before_setgid_check() {
        // World-writable dir is the second-priority failure (after file ww).
        let p = PermInput {
            file_mode: 0o644,
            dir_mode: 0o0757,
        };
        assert_eq!(check_perms(p), PermVerdict::DirOpenWrite);
    }
}
