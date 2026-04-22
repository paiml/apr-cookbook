//! Popperian Falsification Test Suite (APR-022)
//!
//! Following Karl Popper's criterion of demarcation, every published claim must be:
//! 1. **Specific**: Quantified with a measurable threshold
//! 2. **Testable**: Executable via automated test OR cited from a reproducible external harness
//! 3. **Refutable**: Clear conditions under which the claim is considered falsified
//!
//! Run: `cargo test --test falsification -- --nocapture`
//!
//! ## In-process claims (exercised by this file)
//!
//! - **F2**: Zero-copy mmap-backed load completes in < 0.1ms p95 (release) for 100 MB models.
//!   Evidence: aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93 measured 0.028 ms.
//!
//! ## Cited external claims (see docs/specifications/components/quality-gates.md)
//!
//! These are measured by dedicated harnesses in sibling repos. The cookbook does not
//! re-run them; it cites them verbatim with source paths so reviewers can reproduce.
//!
//! - **N1**: Decode ≥ 270 tok/s at c=1 on RTX 4090 with GGUF Q4_K_M.
//!   Source: candle-vs-apr/performance.md:85 (measured 273.8 tok/s).
//! - **N2**: Batch scaling ≥ 10× from c=1 to c=32 for the v5 batch scheduler.
//!   Source: candle-vs-apr/performance.md:150 (measured 13.4×).
//! - **N3**: Load-time parity across APR/GGUF/SafeTensors within 1.5× on the same model.
//!   Source: aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93.
//! - **N4**: Decode advantage vs Candle ≥ 1.15× on identical hardware.
//!   Source: candle-vs-apr/performance.md:85 (measured 1.20×).
//!
//! ## Deleted claims (no evidence found in any repo — see PR #? refactor/aprender-monorepo-v31)
//!
//! These were previously asserted with proxy kernels or synthetic fixtures. They have been
//! removed rather than left as fiction. Any future re-introduction MUST land a real harness
//! and a committed measurement artefact alongside the code.
//!
//! - F1: LZ4 decompression ≥ 3 GB/s — no LZ4 throughput bench in trueno / aprender-compute.
//! - F3: Int4 NMSE < 2% — quantization benches measure throughput only, never accuracy delta.
//! - F4: AES-256-GCM ≥ 100 MB/s — previously used BLAKE3 as a proxy; no crypto bench anywhere.
//! - F5: Whisper WER < 10% — threshold is defined in whisper.apr/THRESHOLDS.md:74 but no
//!   measured WER is logged; previous test simulated WER on hand-written strings, not audio.
//! - F6: FlashAttention ≥ 2× — CPU tiled attention passes 1.0× the naive baseline; the ≥ 2×
//!   claim requires a GPU harness that the cookbook does not host.
//! - F7: AVX-512 ≥ 80 GFLOPS — trueno's SDE infrastructure exists but no published numbers.
//!
//! Reference: Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.

// memmap2::Mmap::map is fundamentally unsafe: the kernel gives us a slice view of
// a file that could be truncated or replaced by another process while we hold the
// mapping. This test accepts that risk in exchange for measuring the real zero-copy
// load path. No safe alternative exists for mmap on Rust.
#![allow(unsafe_code)]

use std::io::Write;
use std::time::{Duration, Instant};

/// Test infrastructure for statistical analysis
mod stats {
    use std::time::Duration;

    /// Calculate percentile from sorted durations.
    pub(crate) fn percentile(sorted: &[Duration], p: f64) -> Duration {
        let idx = ((sorted.len() as f64) * p / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Calculate mean duration.
    pub(crate) fn mean(durations: &[Duration]) -> Duration {
        let total: Duration = durations.iter().sum();
        total / durations.len() as u32
    }
}

// =============================================================================
// F2: Zero-Copy Model Loading Latency (mmap-backed)
// =============================================================================

/// F2: Zero-copy mmap-backed load completes in < 0.1 ms p95 for 100 MB models.
///
/// **Claim**: Loading an mmap-backed `BundledModel` from disk is zero-copy and completes
/// within < 0.1 ms at the 95th percentile, for release builds.
/// **Refutation**: If p95 latency > 0.2 ms in release mode (or > 10 ms in debug), falsified.
///
/// Evidence basis: `aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93` measured
/// 0.028 ms (APR), 0.024 ms (GGUF), 0.029 ms (SafeTensors) on the same test machine.
///
/// Implementation: write a realistic payload to a tempfile, mmap it, and measure the
/// `BundledModel::from_bytes` call on the mapped region. This exercises the real
/// kernel page-mapping path (not the mocked `Vec<u8>` path from prior versions of
/// this test).
#[test]
fn f2_zero_copy_loading_latency() {
    use apr_cookbook::prelude::*;
    use memmap2::Mmap;
    use tempfile::NamedTempFile;

    // Build a realistic-size payload (10 MiB). 100 MiB would be more faithful to the
    // claim's nominal target but inflates CI time; the mmap path is size-insensitive
    // at the system-call layer, so 10 MiB is a representative probe.
    let payload: Vec<u8> = (0..10 * 1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let bundle = ModelBundle::new()
        .with_name("f2-mmap-probe")
        .with_payload(payload)
        .build();

    let mut tmp = NamedTempFile::new().expect("tempfile");
    tmp.write_all(&bundle).expect("write bundle");
    tmp.flush().expect("flush");

    let file = tmp.reopen().expect("reopen tempfile for mmap");
    let mmap = unsafe { Mmap::map(&file).expect("mmap tempfile") };

    // Warmup — first-touch page faults land here, not in measurement.
    for _ in 0..10 {
        let _ = BundledModel::from_bytes(&mmap[..]);
    }

    // Measure: 1000 iterations, capture latencies, sort, take p50/p95/p99.
    let mut latencies = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        let model = BundledModel::from_bytes(&mmap[..]).expect("from_bytes on mmap");
        latencies.push(start.elapsed());
        std::hint::black_box(model);
    }

    latencies.sort();
    let mean = stats::mean(&latencies);
    let p50 = stats::percentile(&latencies, 50.0);
    let p95 = stats::percentile(&latencies, 95.0);
    let p99 = stats::percentile(&latencies, 99.0);

    println!("F2: mmap-backed BundledModel::from_bytes latency (10 MiB payload)");
    println!("F2:   mean: {mean:?}");
    println!("F2:   p50:  {p50:?}");
    println!("F2:   p95:  {p95:?}");
    println!("F2:   p99:  {p99:?}");

    // Release target: < 0.1 ms (measured 0.028 ms in FORMAT_PARITY_REPORT). Debug
    // builds run unoptimized bounds checks and allocator paths that inflate p95 by
    // ~100×, so we hold the debug threshold at 10 ms to avoid flakiness while still
    // catching catastrophic regression.
    let threshold = if cfg!(debug_assertions) {
        Duration::from_millis(10)
    } else {
        Duration::from_micros(200) // 0.2 ms — 2× the release target, per §F2
    };

    assert!(
        p95 < threshold,
        "FALSIFIED: F2 mmap-backed load p95 {p95:?} exceeds {threshold:?} threshold \
         (release target < 0.1 ms, FORMAT_PARITY_REPORT.md:88 measured 0.028 ms)"
    );
}

// =============================================================================
// Meta-Tests: Verify Falsification Infrastructure
// =============================================================================

/// Only claims whose tests live in this file are listed here. Cited external claims
/// (N1–N4) do not have in-process tests — they are validated in sibling harnesses.
const IN_PROCESS_CLAIMS: &[&str] = &["F2"];

/// Verify each in-process claim has a corresponding test function.
#[test]
fn meta_all_claims_have_f_codes() {
    let source = include_str!("falsification.rs");
    for code in IN_PROCESS_CLAIMS {
        let fn_name = format!("fn {}_", code.to_lowercase());
        assert!(
            source.contains(&fn_name),
            "META: claim {code} has no matching test function `{fn_name}…`"
        );
    }
    println!(
        "META: all {} in-process claims have tests",
        IN_PROCESS_CLAIMS.len()
    );
}

/// Verify each in-process claim's test includes an explicit `FALSIFIED:` message.
#[test]
fn meta_all_claims_have_refutation_thresholds() {
    let source = include_str!("falsification.rs");
    for code in IN_PROCESS_CLAIMS {
        let fn_tag = format!("fn {}_", code.to_lowercase());
        let fn_start = source
            .find(&fn_tag)
            .unwrap_or_else(|| panic!("claim {code} missing test fn"));
        let fn_end = source[fn_start..]
            .find("\n}\n")
            .map_or(source.len(), |off| fn_start + off);
        let body = &source[fn_start..fn_end];
        assert!(
            body.contains("FALSIFIED:"),
            "META: {code} test body lacks explicit `FALSIFIED:` refutation message"
        );
    }
    println!("META: all in-process tests carry explicit refutation conditions");
}

/// Cited external claims (N1–N4) are documented in this file's module header. Verify
/// the doc comment references each one with an exact source path so reviewers can
/// reproduce without hunting.
#[test]
fn meta_cited_claims_have_source_paths() {
    let source = include_str!("falsification.rs");
    let cited = [
        ("N1", "candle-vs-apr/performance.md:85"),
        ("N2", "candle-vs-apr/performance.md:150"),
        (
            "N3",
            "aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93",
        ),
        ("N4", "candle-vs-apr/performance.md:85"),
    ];
    for (code, path) in cited {
        assert!(
            source.contains(&format!("**{code}**")) && source.contains(path),
            "META: cited claim {code} missing or lacks source path `{path}`"
        );
    }
    println!("META: all cited claims have reproducible source paths");
}
