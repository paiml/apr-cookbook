//! # apr ptx --strict — Instruction Whitelist Enforcement
//!
//! `apr ptx <FILE> --strict` flags any non-whitelisted PTX instruction.
//! The whitelist covers FP16/BF16 math, common vector ops, ld/st, and
//! sync barriers. Anything outside (e.g., legacy ATOM_CAS_64, debug
//! stmts) becomes a violation. This recipe codifies the whitelist
//! check.
//!
//! Demonstrates the **PTX.7** recipe for PMAT-111 (apr ptx coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-001 + PTX ISA 8.5
//!
//! Run with: cargo run --example cli_ptx_strict_mode_whitelist
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const WHITELIST: &[&str] = &[
    "ld.global",
    "st.global",
    "ld.shared",
    "st.shared",
    "mov",
    "add",
    "mul",
    "fma",
    "cvt",
    "selp",
    "setp",
    "bar.sync",
    "bar.warp.sync",
    "ldg",
    "ld.cv",
    "wmma",
    "mma",
];

#[derive(Debug, PartialEq)]
pub enum InstrVerdict {
    Allowed,
    Denied { matched_prefix: bool },
}

pub fn check_instruction(instr: &str, strict: bool) -> InstrVerdict {
    if !strict {
        return InstrVerdict::Allowed;
    }
    for w in WHITELIST {
        if instr == *w || instr.starts_with(&format!("{w}.")) {
            return InstrVerdict::Allowed;
        }
    }
    let matched_prefix = WHITELIST.iter().any(|w| {
        let prefix = w.split('.').next().unwrap_or("");
        !prefix.is_empty() && instr.starts_with(prefix)
    });
    InstrVerdict::Denied { matched_prefix }
}

pub fn count_violations(instrs: &[&str]) -> usize {
    instrs
        .iter()
        .filter(|i| matches!(check_instruction(i, true), InstrVerdict::Denied { .. }))
        .count()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_strict_mode_whitelist")?;

    let kernel = [
        "ld.global.v4.f32",
        "fma.rn.f32",
        "atom.global.cas.b64", // not whitelisted
        "bar.sync",
        "vote.ballot.sync", // not whitelisted
        "st.global.v4.f32",
    ];

    for i in kernel {
        println!("{i:<28} → {:?}", check_instruction(i, true));
    }
    println!("Violations: {}", count_violations(&kernel));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitelist_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn whitelisted_instruction_allowed() {
        assert_eq!(check_instruction("fma", true), InstrVerdict::Allowed);
    }

    #[test]
    fn whitelisted_with_modifier_allowed() {
        // `fma.rn.f32` matches `fma` family.
        assert_eq!(check_instruction("fma.rn.f32", true), InstrVerdict::Allowed);
    }

    #[test]
    fn unknown_instruction_denied() {
        let v = check_instruction("atom.global.cas.b64", true);
        assert!(matches!(v, InstrVerdict::Denied { .. }));
    }

    #[test]
    fn non_strict_allows_everything() {
        // Without --strict, even unknown instructions pass.
        assert_eq!(
            check_instruction("vote.ballot.sync", false),
            InstrVerdict::Allowed
        );
    }

    #[test]
    fn ld_shared_whitelisted_for_smem() {
        assert_eq!(
            check_instruction("ld.shared.f32", true),
            InstrVerdict::Allowed
        );
    }

    #[test]
    fn count_violations_in_kernel() {
        let kernel = [
            "ld.global.f32",
            "atom.shared.cas.b32",
            "vote.any.sync",
            "fma.rn.f32",
        ];
        assert_eq!(count_violations(&kernel), 2);
    }

    #[test]
    fn empty_kernel_zero_violations() {
        assert_eq!(count_violations(&[]), 0);
    }
}
