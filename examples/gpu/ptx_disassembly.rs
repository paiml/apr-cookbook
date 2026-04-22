//! # Recipe: PTX Kernel Disassembly Reader
//!
//! **Category**: gpu
//! **CLI Equivalent**: `apr ptx disassemble kernel.cubin --format text`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example ptx_disassembly` exits 0
//! 2. [x] `cargo test --example ptx_disassembly` passes
//! 3. [x] Deterministic output (no RNG needed)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr ptx disassemble` in-process (no shell-out)
//! 10. [x] Unit tests cover directive parsing, instruction count, kernel header
//!
//! ## Learning Objective
//! Demonstrates a PTX disassembly reader: parses a synthetic .ptx string into
//! directives, kernel entrypoints, and instruction mnemonics, then renders a
//! structured listing with a per-kernel instruction histogram. Mirrors the
//! output `apr ptx disassemble` produces from compiled cubins.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ptx_disassembly
//! ```
//!
//! ## References
//! - Lattner, C. et al. (2021). *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*. CGO. arXiv:2002.11054

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtxLine {
    Directive(String),
    KernelHeader(String),
    Instruction { mnemonic: String, full: String },
    Other(String),
}

pub fn parse_ptx(src: &str) -> Vec<PtxLine> {
    src.lines()
        .map(|l| {
            let trimmed = l.trim();
            if trimmed.starts_with(".version")
                || trimmed.starts_with(".target")
                || trimmed.starts_with(".address_size")
            {
                PtxLine::Directive(trimmed.into())
            } else if trimmed.starts_with(".visible .entry") || trimmed.starts_with(".entry") {
                PtxLine::KernelHeader(trimmed.into())
            } else if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('{')
                || trimmed.starts_with('}')
            {
                PtxLine::Other(trimmed.into())
            } else {
                // First token (before whitespace, '.', or ';') is the mnemonic.
                let mnemonic = trimmed
                    .split(|c: char| c.is_whitespace() || c == '.' || c == ';')
                    .find(|s| !s.is_empty())
                    .unwrap_or("")
                    .to_string();
                if mnemonic.is_empty() {
                    PtxLine::Other(trimmed.into())
                } else {
                    PtxLine::Instruction {
                        mnemonic,
                        full: trimmed.to_string(),
                    }
                }
            }
        })
        .collect()
}

pub fn instruction_histogram(lines: &[PtxLine]) -> BTreeMap<String, u32> {
    let mut hist = BTreeMap::new();
    for l in lines {
        if let PtxLine::Instruction { mnemonic, .. } = l {
            *hist.entry(mnemonic.clone()).or_insert(0) += 1;
        }
    }
    hist
}

pub fn kernel_names(lines: &[PtxLine]) -> Vec<String> {
    lines
        .iter()
        .filter_map(|l| {
            if let PtxLine::KernelHeader(h) = l {
                h.split_whitespace()
                    .last()
                    .map(|s| s.trim_end_matches('(').to_string())
            } else {
                None
            }
        })
        .collect()
}

const DEMO_PTX: &str = ".version 7.5
.target sm_80
.address_size 64

.visible .entry gemm_kernel(
    .param .u64 ptr_a,
    .param .u64 ptr_b
)
{
    ld.param.u64 %rd1, [ptr_a];
    ld.param.u64 %rd2, [ptr_b];
    mad.lo.s32 %r1, %r2, %r3, %r4;
    st.global.f32 [%rd1], %f1;
    ret;
}
";

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ptx_disassembly")?;
    println!("=== Recipe: {} ===", ctx.name());

    let lines = parse_ptx(DEMO_PTX);
    for l in &lines {
        match l {
            PtxLine::Directive(d) => println!("  DIRECTIVE    {}", d),
            PtxLine::KernelHeader(h) => println!("  KERNEL-HEAD  {}", h),
            PtxLine::Instruction { mnemonic, full } => {
                println!("  INSTR {:<10} {}", mnemonic, full);
            }
            PtxLine::Other(o) => println!("  OTHER        {}", o),
        }
    }

    let hist = instruction_histogram(&lines);
    let kernels = kernel_names(&lines);
    println!("Kernels: {:?}", kernels);
    println!("Instruction histogram:");
    for (m, c) in &hist {
        println!("  {:<12} {}", m, c);
    }

    let report = json!({
        "recipe": ctx.name(),
        "kernels": kernels,
        "instruction_count": lines.iter().filter(|l| matches!(l, PtxLine::Instruction { .. })).count(),
        "histogram": hist,
    });
    let path = ctx.path("ptx-disassembly.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("kernels", kernels.len() as i64);
    ctx.record_metric(
        "instructions",
        lines
            .iter()
            .filter(|l| matches!(l, PtxLine::Instruction { .. }))
            .count() as i64,
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_directives() {
        let lines = parse_ptx(".version 7.5\n.target sm_80");
        assert_eq!(lines[0], PtxLine::Directive(".version 7.5".into()));
        assert_eq!(lines[1], PtxLine::Directive(".target sm_80".into()));
    }

    #[test]
    fn parses_kernel_header() {
        let lines = parse_ptx(".visible .entry foo()");
        assert!(matches!(lines[0], PtxLine::KernelHeader(_)));
    }

    #[test]
    fn histogram_counts_mnemonics() {
        let lines = parse_ptx("ld.param.u64 %rd1, x;\nld.param.u64 %rd2, y;\nret;");
        let hist = instruction_histogram(&lines);
        assert_eq!(hist.get("ld"), Some(&2));
        assert_eq!(hist.get("ret"), Some(&1));
    }

    #[test]
    fn kernel_names_extracted() {
        let lines = parse_ptx(".visible .entry gemm(");
        let names = kernel_names(&lines);
        assert!(names.iter().any(|n| n.contains("gemm")));
    }

    #[test]
    fn blank_lines_are_other() {
        let lines = parse_ptx("\n{\n}\n");
        assert!(lines.iter().all(|l| matches!(l, PtxLine::Other(_))));
    }
}
