//! # WASM WASI Capability Grant
//!
//! WASI-preview1 follows capability-based security: the host grants the
//! guest a set of preopen FDs (filesystem dirs) and env vars; the guest
//! can ONLY access those. This recipe builds the grant validator —
//! reject paths outside preopens, normalize redundant slashes, and
//! flag escape attempts (`..`).
//!
//! Demonstrates the **WASM.13** recipe for PMAT-134 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WASI capability-based security model.
//!
//! Run with: cargo run --example wasm_wasi_capability_grant
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GrantVerdict {
    Ok { canonical_path: String },
    OutsidePreopens,
    EscapeAttempt,
    EmptyPreopens,
    EmptyPath,
}

pub fn check(path: &str, preopens: &[&str]) -> GrantVerdict {
    if path.is_empty() {
        return GrantVerdict::EmptyPath;
    }
    if preopens.is_empty() {
        return GrantVerdict::EmptyPreopens;
    }
    if path.split('/').any(|seg| seg == "..") {
        return GrantVerdict::EscapeAttempt;
    }
    let canonical = canonicalize(path);
    let granted = preopens
        .iter()
        .any(|p| canonical == *p || canonical.starts_with(&format!("{p}/")));
    if !granted {
        return GrantVerdict::OutsidePreopens;
    }
    GrantVerdict::Ok {
        canonical_path: canonical,
    }
}

fn canonicalize(path: &str) -> String {
    let trimmed: String = path
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect::<Vec<_>>()
        .join("/");
    if path.starts_with('/') {
        format!("/{trimmed}")
    } else {
        trimmed
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_wasi_capability_grant")?;

    let preopens = ["/data/models", "/tmp"];
    for p in [
        "/data/models/llama.gguf",
        "/data/models//x/y",
        "/etc/passwd",
        "/data/models/../../../etc/passwd",
        "/tmp/scratch.bin",
    ] {
        println!("{p:<40}  →  {:?}", check(p, &preopens));
    }
    println!("empty path: {:?}", check("", &preopens));
    println!("empty preopens: {:?}", check("/x", &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn path_inside_preopen_granted() {
        let v = check("/data/models/llama.gguf", &["/data/models"]);
        assert!(matches!(v, GrantVerdict::Ok { .. }));
    }

    #[test]
    fn path_outside_preopen_rejected() {
        let v = check("/etc/passwd", &["/data/models"]);
        assert_eq!(v, GrantVerdict::OutsidePreopens);
    }

    #[test]
    fn escape_attempt_rejected() {
        let v = check("/data/models/../../../etc/passwd", &["/data/models"]);
        assert_eq!(v, GrantVerdict::EscapeAttempt);
    }

    #[test]
    fn empty_path_rejected() {
        assert_eq!(check("", &["/data"]), GrantVerdict::EmptyPath);
    }

    #[test]
    fn empty_preopens_rejected() {
        assert_eq!(check("/x", &[]), GrantVerdict::EmptyPreopens);
    }

    #[test]
    fn redundant_slashes_normalized() {
        let v = check("/data/models//x/y", &["/data/models"]);
        if let GrantVerdict::Ok { canonical_path } = v {
            assert_eq!(canonical_path, "/data/models/x/y");
        }
    }

    #[test]
    fn current_dir_dot_normalized_away() {
        let v = check("/data/models/./x", &["/data/models"]);
        if let GrantVerdict::Ok { canonical_path } = v {
            assert_eq!(canonical_path, "/data/models/x");
        }
    }

    #[test]
    fn second_preopen_reachable() {
        let v = check("/tmp/scratch.bin", &["/data/models", "/tmp"]);
        assert!(matches!(v, GrantVerdict::Ok { .. }));
    }

    #[test]
    fn exact_preopen_root_granted() {
        let v = check("/data/models", &["/data/models"]);
        assert!(matches!(v, GrantVerdict::Ok { .. }));
    }

    #[test]
    fn similar_prefix_not_substring_match() {
        // "/data/models2" should NOT be granted just because preopen is "/data/models".
        let v = check("/data/models2/x", &["/data/models"]);
        assert_eq!(v, GrantVerdict::OutsidePreopens);
    }

    #[test]
    fn dotdot_anywhere_in_path_rejected() {
        let v = check("/data/foo/../bar", &["/data"]);
        assert_eq!(v, GrantVerdict::EscapeAttempt);
    }
}
