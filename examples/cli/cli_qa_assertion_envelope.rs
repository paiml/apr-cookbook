//! # apr qa — Assertion Envelope (CI Threshold Composition)
//!
//! `apr qa <FILE> --assert-tps <N> --assert-speedup <X> --assert-gpu-speedup <Y>`
//! composes a falsifiable QA checklist for model releases. This recipe
//! models the assertion envelope (CLI args → check list → exit code) so
//! a CI pipeline can preview which checks will run and what they'll
//! enforce before invoking the binary.
//!
//! Demonstrates the **QA.3** recipe for PMAT-093 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender F-PERF-042 + sysexits.h conventions
//!
//! Run with: cargo run --example cli_qa_assertion_envelope
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QaCheck {
    GoldenOutput,
    Throughput { min_tps: f64 },
    OllamaSpeedup { min_speedup: f64 },
    GpuSpeedup { min_speedup: f64 },
    Contract,
    FormatParity,
    PtxParity,
}

#[derive(Debug, Default, Clone)]
pub struct QaFlags {
    pub assert_tps: Option<f64>,
    pub assert_speedup: Option<f64>,
    pub assert_gpu_speedup: Option<f64>,
    pub skip_golden: bool,
    pub skip_throughput: bool,
    pub skip_ollama: bool,
    pub skip_gpu_speedup: bool,
    pub skip_contract: bool,
    pub skip_format_parity: bool,
    pub skip_ptx_parity: bool,
}

pub fn build_check_list(flags: &QaFlags) -> Vec<QaCheck> {
    let mut out = Vec::new();
    if !flags.skip_golden {
        out.push(QaCheck::GoldenOutput);
    }
    if !flags.skip_throughput {
        out.push(QaCheck::Throughput {
            min_tps: flags.assert_tps.unwrap_or(0.0),
        });
    }
    if !flags.skip_ollama {
        out.push(QaCheck::OllamaSpeedup {
            min_speedup: flags.assert_speedup.unwrap_or(0.0),
        });
    }
    if !flags.skip_gpu_speedup {
        out.push(QaCheck::GpuSpeedup {
            min_speedup: flags.assert_gpu_speedup.unwrap_or(0.0),
        });
    }
    if !flags.skip_contract {
        out.push(QaCheck::Contract);
    }
    if !flags.skip_format_parity {
        out.push(QaCheck::FormatParity);
    }
    if !flags.skip_ptx_parity {
        out.push(QaCheck::PtxParity);
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_assertion_envelope")?;

    let strict = QaFlags {
        assert_tps: Some(50.0),
        assert_speedup: Some(2.0),
        assert_gpu_speedup: Some(5.0),
        ..Default::default()
    };
    let cpu_only = QaFlags {
        assert_tps: Some(20.0),
        skip_gpu_speedup: true,
        skip_ptx_parity: true,
        skip_ollama: true,
        ..Default::default()
    };
    let smoke = QaFlags {
        skip_golden: false,
        skip_throughput: true,
        skip_ollama: true,
        skip_gpu_speedup: true,
        skip_format_parity: true,
        skip_ptx_parity: true,
        ..Default::default()
    };

    println!(
        "strict checks ({}):     {:?}",
        build_check_list(&strict).len(),
        build_check_list(&strict)
    );
    println!(
        "cpu-only checks ({}):   {:?}",
        build_check_list(&cpu_only).len(),
        build_check_list(&cpu_only)
    );
    println!(
        "smoke checks ({}):      {:?}",
        build_check_list(&smoke).len(),
        build_check_list(&smoke)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_flags_run_all_seven_checks() {
        let checks = build_check_list(&QaFlags::default());
        assert_eq!(checks.len(), 7);
    }

    #[test]
    fn skip_flags_remove_their_check() {
        let flags = QaFlags {
            skip_golden: true,
            skip_contract: true,
            ..Default::default()
        };
        let checks = build_check_list(&flags);
        assert!(!checks.iter().any(|c| matches!(c, QaCheck::GoldenOutput)));
        assert!(!checks.iter().any(|c| matches!(c, QaCheck::Contract)));
        assert_eq!(checks.len(), 5);
    }

    #[test]
    fn assert_tps_threaded_into_throughput_check() {
        let flags = QaFlags {
            assert_tps: Some(75.0),
            ..Default::default()
        };
        let checks = build_check_list(&flags);
        let tps_check = checks
            .iter()
            .find(|c| matches!(c, QaCheck::Throughput { .. }));
        assert!(matches!(
            tps_check,
            Some(QaCheck::Throughput { min_tps }) if (*min_tps - 75.0).abs() < 1e-9
        ));
    }

    #[test]
    fn skip_all_yields_empty_check_list() {
        // Pathological config — all skips set. Allowed (caller's choice) but
        // the binary will then exit 0 trivially.
        let flags = QaFlags {
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            ..Default::default()
        };
        assert!(build_check_list(&flags).is_empty());
    }
}
