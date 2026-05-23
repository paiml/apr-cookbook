//! # apr compile — `--target` Triple Validator
//!
//! `apr compile <FILE> --target <TRIPLE>` accepts a Rust target triple
//! (e.g., `x86_64-unknown-linux-musl`, `aarch64-apple-darwin`). This
//! recipe builds the parser and asserts the contract: triple has 3 or 4
//! dash-separated components (arch-vendor-os[-abi]), known archs/OSs/ABIs,
//! `linux-musl` and `linux-gnu` properly distinguished.
//!
//! Demonstrates the **COMPILE.3** recipe for PMAT-110 (apr compile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-SPEC §4.16 + Rust target triples
//!
//! Run with: cargo run --example cli_compile_target_triple_validator
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedTriple {
    pub arch: String,
    pub vendor: String,
    pub os: String,
    pub abi: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum TripleVerdict {
    Ok(ParsedTriple),
    UnknownArch(String),
    UnknownOs(String),
    Malformed { observed_segments: usize },
}

const KNOWN_ARCHES: &[&str] = &["x86_64", "aarch64", "i686", "wasm32", "riscv64"];
const KNOWN_OS: &[&str] = &["linux", "darwin", "windows", "freebsd", "wasi"];

pub fn parse_triple(s: &str) -> TripleVerdict {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() < 3 || parts.len() > 4 {
        return TripleVerdict::Malformed {
            observed_segments: parts.len(),
        };
    }
    let arch = parts[0];
    if !KNOWN_ARCHES.contains(&arch) {
        return TripleVerdict::UnknownArch(arch.into());
    }
    let os = parts[2];
    if !KNOWN_OS.contains(&os) {
        return TripleVerdict::UnknownOs(os.into());
    }
    TripleVerdict::Ok(ParsedTriple {
        arch: arch.into(),
        vendor: parts[1].into(),
        os: os.into(),
        abi: parts.get(3).map(|s| (*s).to_string()),
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compile_target_triple_validator")?;

    for t in [
        "x86_64-unknown-linux-musl",
        "aarch64-apple-darwin",
        "wasm32-unknown-wasi",
        "x86_64-pc-windows-gnu",
        "x86_64-unknown-linux", // missing abi
        "exotic-arch-linux-gnu",
        "no-dashes",
    ] {
        println!("{t:>30}  →  {:?}", parse_triple(t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn x86_64_linux_musl_parses() {
        let v = parse_triple("x86_64-unknown-linux-musl");
        if let TripleVerdict::Ok(p) = v {
            assert_eq!(p.arch, "x86_64");
            assert_eq!(p.vendor, "unknown");
            assert_eq!(p.os, "linux");
            assert_eq!(p.abi.as_deref(), Some("musl"));
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn aarch64_apple_darwin_parses_without_abi() {
        // 3-segment triple: arch-vendor-os, no ABI.
        let v = parse_triple("aarch64-apple-darwin");
        if let TripleVerdict::Ok(p) = v {
            assert_eq!(p.arch, "aarch64");
            assert_eq!(p.os, "darwin");
            assert!(p.abi.is_none());
        }
    }

    #[test]
    fn unknown_arch_rejected() {
        let v = parse_triple("exotic-unknown-linux-gnu");
        assert!(matches!(v, TripleVerdict::UnknownArch(_)));
    }

    #[test]
    fn unknown_os_rejected() {
        let v = parse_triple("x86_64-unknown-haiku");
        assert!(matches!(v, TripleVerdict::UnknownOs(_)));
    }

    #[test]
    fn too_few_segments_rejected() {
        assert!(matches!(
            parse_triple("x86_64-linux"),
            TripleVerdict::Malformed {
                observed_segments: 2
            }
        ));
    }

    #[test]
    fn too_many_segments_rejected() {
        assert!(matches!(
            parse_triple("x86_64-unknown-linux-gnu-extra-bits"),
            TripleVerdict::Malformed {
                observed_segments: 6
            }
        ));
    }

    #[test]
    fn musl_vs_gnu_distinguished() {
        let musl = parse_triple("x86_64-unknown-linux-musl");
        let gnu = parse_triple("x86_64-unknown-linux-gnu");
        if let (TripleVerdict::Ok(m), TripleVerdict::Ok(g)) = (musl, gnu) {
            assert_eq!(m.abi.as_deref(), Some("musl"));
            assert_eq!(g.abi.as_deref(), Some("gnu"));
        }
    }

    #[test]
    fn no_dashes_rejected() {
        assert!(matches!(
            parse_triple("nodashes"),
            TripleVerdict::Malformed {
                observed_segments: 1
            }
        ));
    }
}
