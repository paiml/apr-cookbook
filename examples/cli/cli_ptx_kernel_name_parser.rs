//! # apr ptx — Kernel Name Parser
//!
//! `apr ptx <FILE>` lists kernels matching `.entry <name>(...)` lines.
//! Real PTX uses Itanium-style mangling for C++ symbols. This recipe
//! builds a tolerant parser that extracts kernel names from `.entry`
//! lines, skipping comments and `.func` (device-only) entries.
//!
//! Demonstrates the **PTX.8** recipe for PMAT-111 (apr ptx coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-001 + PTX ISA 8.5 §3.1
//!
//! Run with: cargo run --example cli_ptx_kernel_name_parser
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub struct KernelEntry {
    pub name: String,
    pub is_visible: bool,
}

pub fn parse_kernels(ptx: &str) -> Vec<KernelEntry> {
    let mut out = Vec::new();
    for raw in ptx.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let is_visible = line.starts_with(".visible .entry") || line.starts_with(".visible.entry");
        let is_entry = is_visible || line.starts_with(".entry") || line.starts_with(".weak .entry");
        if !is_entry {
            continue;
        }
        if let Some(name) = extract_name(line) {
            out.push(KernelEntry {
                name: name.to_string(),
                is_visible,
            });
        }
    }
    out
}

fn strip_comment(line: &str) -> &str {
    if let Some(idx) = line.find("//") {
        &line[..idx]
    } else {
        line
    }
}

fn extract_name(line: &str) -> Option<&str> {
    // After `.entry`, the name token comes next.
    let after_entry = line.find(".entry")? + ".entry".len();
    let rest = line.get(after_entry..)?.trim_start();
    let end = rest.find(['(', ' ', '\t']).unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_kernel_name_parser")?;

    let sample = "\
        // a comment\n\
        .version 8.5\n\
        .target sm_90\n\
        .visible .entry fwd_pass(\n\
            .param .u64 input,\n\
            .param .u64 output\n\
        )\n\
        .entry _Z9bwd_passPfS_(.param .u64 a, .param .u64 b)\n\
        .func helper() // device-only, skipped\n\
    ";
    for k in parse_kernels(sample) {
        println!("{} (visible={})", k.name, k.is_visible);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn finds_visible_entry() {
        let p = ".visible .entry add_kernel(.param .u64 a)";
        let k = parse_kernels(p);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "add_kernel");
        assert!(k[0].is_visible);
    }

    #[test]
    fn finds_plain_entry_as_internal() {
        let p = ".entry internal_helper(.param .u64 a)";
        let k = parse_kernels(p);
        assert_eq!(k.len(), 1);
        assert!(!k[0].is_visible);
    }

    #[test]
    fn ignores_func_definitions() {
        // `.func` is device-only; not a kernel.
        let p = ".func helper() {}";
        assert!(parse_kernels(p).is_empty());
    }

    #[test]
    fn ignores_comments() {
        let p = "// .entry not_a_kernel()";
        assert!(parse_kernels(p).is_empty());
    }

    #[test]
    fn parses_mangled_cpp_names() {
        let p = ".entry _Z9bwd_passPfS_(.param .u64 a)";
        let k = parse_kernels(p);
        assert_eq!(k[0].name, "_Z9bwd_passPfS_");
    }

    #[test]
    fn handles_multiple_kernels() {
        let p = "\
            .visible .entry k1()\n\
            .visible .entry k2()\n\
            .entry k3()\n\
        ";
        let k = parse_kernels(p);
        assert_eq!(k.len(), 3);
        assert_eq!(k[0].name, "k1");
        assert_eq!(k[2].name, "k3");
    }

    #[test]
    fn empty_input_yields_no_kernels() {
        assert!(parse_kernels("").is_empty());
        assert!(parse_kernels("\n\n  \n").is_empty());
    }

    #[test]
    fn strips_inline_comment_after_entry() {
        let p = ".visible .entry foo() // trailing comment";
        let k = parse_kernels(p);
        assert_eq!(k[0].name, "foo");
    }
}
