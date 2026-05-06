//! # apr compile — `--output` Path Resolver
//!
//! `apr compile <FILE> -o <OUT>` derives a default output path from
//! the model name when `-o` is omitted: strip `.apr` extension, append
//! target-specific extension (`.exe` for windows, none otherwise). This
//! recipe builds the resolver and asserts the contract.
//!
//! Demonstrates the **COMPILE.5** recipe for PMAT-110 (apr compile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-SPEC §4.16
//!
//! Run with: cargo run --example cli_compile_output_path_resolver
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::path::PathBuf;

pub fn resolve_output_path(input: &str, target_os: &str, explicit: Option<&str>) -> PathBuf {
    if let Some(p) = explicit {
        return PathBuf::from(p);
    }
    let stem = std::path::Path::new(input)
        .file_stem()
        .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
    let ext = match target_os {
        "windows" => ".exe",
        _ => "",
    };
    PathBuf::from(format!("{stem}{ext}"))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compile_output_path_resolver")?;

    let cases = [
        ("model.apr", "linux", None),
        ("model.apr", "windows", None),
        ("model.apr", "linux", Some("/tmp/custom-out")),
        ("/path/to/llama.apr", "darwin", None),
        ("noext", "linux", None),
    ];
    for (input, os, explicit) in cases {
        let p = resolve_output_path(input, os, explicit);
        println!("({input:>20}, {os:>7}, {explicit:?})  →  {}", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn explicit_output_passes_through() {
        let p = resolve_output_path("model.apr", "linux", Some("/tmp/foo"));
        assert_eq!(p, PathBuf::from("/tmp/foo"));
    }

    #[test]
    fn linux_no_extension() {
        let p = resolve_output_path("model.apr", "linux", None);
        assert_eq!(p, PathBuf::from("model"));
    }

    #[test]
    fn windows_appends_exe() {
        let p = resolve_output_path("model.apr", "windows", None);
        assert_eq!(p, PathBuf::from("model.exe"));
    }

    #[test]
    fn darwin_no_extension() {
        let p = resolve_output_path("model.apr", "darwin", None);
        assert_eq!(p, PathBuf::from("model"));
    }

    #[test]
    fn full_path_strips_directory() {
        let p = resolve_output_path("/path/to/llama.apr", "linux", None);
        assert_eq!(p, PathBuf::from("llama"));
    }

    #[test]
    fn extensionless_input_keeps_full_name() {
        let p = resolve_output_path("noext", "linux", None);
        assert_eq!(p, PathBuf::from("noext"));
    }

    #[test]
    fn empty_input_falls_back_to_model() {
        let p = resolve_output_path("", "linux", None);
        assert_eq!(p, PathBuf::from("model"));
    }

    #[test]
    fn windows_explicit_does_not_auto_append_exe() {
        // If operator gave explicit path, respect it verbatim.
        let p = resolve_output_path("model.apr", "windows", Some("custom"));
        assert_eq!(p, PathBuf::from("custom"));
    }
}
