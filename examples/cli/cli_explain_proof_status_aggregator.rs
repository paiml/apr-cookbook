//! # apr explain — `--proof-status` Per-Kernel Aggregator
//!
//! `apr explain --kernel <ARCH> --proof-status` shows the contract proof
//! status for each kernel: `Proved` / `WIP` / `Sorry` / `NotApplicable`.
//! This recipe builds the per-architecture aggregator and asserts the
//! contract: rollup uses worst-of severity (Sorry > WIP > Proved >
//! NotApplicable), and the report is sorted by kernel name for
//! deterministic CI logs.
//!
//! Demonstrates the **EXPLAIN.6** recipe for PMAT-099 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender Lean-scaffold proof-status convention
//!
//! Run with: cargo run --example cli_explain_proof_status_aggregator
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofStatus {
    NotApplicable,
    Proved,
    Wip,
    Sorry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelProof {
    pub kernel: String,
    pub status: ProofStatus,
}

pub fn rollup_status(proofs: &[KernelProof]) -> ProofStatus {
    proofs
        .iter()
        .map(|p| p.status)
        .max()
        .unwrap_or(ProofStatus::NotApplicable)
}

pub fn aggregate_by_status(proofs: &[KernelProof]) -> BTreeMap<ProofStatus, Vec<String>> {
    let mut out: BTreeMap<ProofStatus, Vec<String>> = BTreeMap::new();
    for p in proofs {
        out.entry(p.status).or_default().push(p.kernel.clone());
    }
    for v in out.values_mut() {
        v.sort();
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_proof_status_aggregator")?;

    let proofs = vec![
        KernelProof {
            kernel: "BF16Embed".into(),
            status: ProofStatus::Proved,
        },
        KernelProof {
            kernel: "BF16RmsNorm".into(),
            status: ProofStatus::Proved,
        },
        KernelProof {
            kernel: "BF16AttnQkv".into(),
            status: ProofStatus::Wip,
        },
        KernelProof {
            kernel: "BF16SwiGLU".into(),
            status: ProofStatus::Sorry,
        },
        KernelProof {
            kernel: "BF16Unembed".into(),
            status: ProofStatus::NotApplicable,
        },
    ];

    println!("rollup: {:?}", rollup_status(&proofs));
    for (status, kernels) in aggregate_by_status(&proofs) {
        println!("  {status:?}:  {kernels:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kp(name: &str, status: ProofStatus) -> KernelProof {
        KernelProof {
            kernel: name.into(),
            status,
        }
    }

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn status_ordering_is_na_lt_proved_lt_wip_lt_sorry() {
        // Sorry is the worst, NotApplicable is the best (effectively "skip").
        assert!(ProofStatus::NotApplicable < ProofStatus::Proved);
        assert!(ProofStatus::Proved < ProofStatus::Wip);
        assert!(ProofStatus::Wip < ProofStatus::Sorry);
    }

    #[test]
    fn rollup_picks_worst_status() {
        let proofs = vec![
            kp("a", ProofStatus::Proved),
            kp("b", ProofStatus::Wip),
            kp("c", ProofStatus::Proved),
        ];
        assert_eq!(rollup_status(&proofs), ProofStatus::Wip);
    }

    #[test]
    fn rollup_with_sorry_returns_sorry() {
        let proofs = vec![
            kp("a", ProofStatus::Proved),
            kp("b", ProofStatus::Sorry),
            kp("c", ProofStatus::Wip),
        ];
        assert_eq!(rollup_status(&proofs), ProofStatus::Sorry);
    }

    #[test]
    fn empty_proofs_rollup_to_not_applicable() {
        assert_eq!(rollup_status(&[]), ProofStatus::NotApplicable);
    }

    #[test]
    fn aggregator_groups_by_status() {
        let proofs = vec![
            kp("a", ProofStatus::Proved),
            kp("b", ProofStatus::Wip),
            kp("c", ProofStatus::Proved),
            kp("d", ProofStatus::Sorry),
        ];
        let agg = aggregate_by_status(&proofs);
        assert_eq!(
            agg[&ProofStatus::Proved],
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(agg[&ProofStatus::Wip], vec!["b".to_string()]);
        assert_eq!(agg[&ProofStatus::Sorry], vec!["d".to_string()]);
    }

    #[test]
    fn aggregator_sorts_kernels_within_each_status() {
        let proofs = vec![
            kp("z", ProofStatus::Proved),
            kp("a", ProofStatus::Proved),
            kp("m", ProofStatus::Proved),
        ];
        let agg = aggregate_by_status(&proofs);
        assert_eq!(
            agg[&ProofStatus::Proved],
            vec!["a".to_string(), "m".to_string(), "z".to_string()]
        );
    }
}
