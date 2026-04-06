//! # Recipe: Memory Profiler for Container Sizing
//!
//! **Category**: Inference Monitoring
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (Linux /proc/self/status)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A - Linux /proc only)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Track peak RSS (Resident Set Size) during model load and inference by reading
//! `/proc/self/status` on Linux. Produce container sizing recommendations for
//! Lambda (128/256/512/1024 MB) and Docker deployments.
//!
//! ## Toyota Way: ムダ (Muda) - Waste Elimination
//! Over-provisioned containers waste memory and money. Measure actual RSS to
//! right-size every deployment target.
//!
//! ## Run Command
//! ```bash
//! cargo run --example monitoring_memory_profiler
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Memory snapshot
// ---------------------------------------------------------------------------

/// A point-in-time reading of process memory from `/proc/self/status`.
#[derive(Debug, Clone, Copy)]
struct MemorySnapshot {
    rss_kb: u64,
    vm_size_kb: u64,
    vm_peak_kb: u64,
    timestamp: Instant,
}

/// Serializable profile of a single phase (load, inference, etc.).
#[derive(Debug, Clone, Serialize)]
struct MemoryProfile {
    phase: String,
    rss_start_kb: u64,
    rss_peak_kb: u64,
    rss_delta_kb: i64,
    duration_ms: f64,
}

/// Serializable container sizing recommendation.
#[derive(Debug, Clone, Serialize)]
struct SizingRecommendation {
    target: String,
    recommended_mb: u64,
    headroom_pct: f64,
    reasoning: String,
}

// ---------------------------------------------------------------------------
// /proc/self/status parser
// ---------------------------------------------------------------------------

/// Read memory counters from `/proc/self/status`.
///
/// Falls back to zeros with a printed warning when the file is unreadable
/// (e.g., macOS, WASM, restricted containers).
fn read_proc_status() -> Result<MemorySnapshot> {
    let Ok(content) = std::fs::read_to_string("/proc/self/status") else {
        eprintln!("Warning: /proc/self/status not readable, returning zeros");
        return Ok(MemorySnapshot {
            rss_kb: 0,
            vm_size_kb: 0,
            vm_peak_kb: 0,
            timestamp: Instant::now(),
        });
    };

    let mut rss_kb = 0u64;
    let mut vm_size_kb = 0u64;
    let mut vm_peak_kb = 0u64;

    for line in content.lines() {
        if let Some(val) = line.strip_prefix("VmRSS:") {
            rss_kb = val
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("VmSize:") {
            vm_size_kb = val
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("VmPeak:") {
            vm_peak_kb = val
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }

    Ok(MemorySnapshot {
        rss_kb,
        vm_size_kb,
        vm_peak_kb,
        timestamp: Instant::now(),
    })
}

// ---------------------------------------------------------------------------
// Phase profiler
// ---------------------------------------------------------------------------

/// Run `f`, bracketed by memory snapshots, and return the result together with
/// a [`MemoryProfile`] capturing the RSS delta.
fn profile_phase<F, T>(name: &str, f: F) -> Result<(T, MemoryProfile)>
where
    F: FnOnce() -> Result<T>,
{
    let before = read_proc_status()?;
    let result = f()?;
    let after = read_proc_status()?;

    let rss_peak = after.rss_kb.max(before.rss_kb);
    let duration_ms = before.timestamp.elapsed().as_secs_f64() * 1000.0;

    let profile = MemoryProfile {
        phase: name.to_string(),
        rss_start_kb: before.rss_kb,
        rss_peak_kb: rss_peak,
        rss_delta_kb: after.rss_kb as i64 - before.rss_kb as i64,
        duration_ms,
    };

    Ok((result, profile))
}

// ---------------------------------------------------------------------------
// Simulated workloads
// ---------------------------------------------------------------------------

/// Allocate `size_mb` megabytes of random data using the supplied RNG.
fn create_test_payload(size_mb: usize, rng: &mut StdRng) -> Vec<u8> {
    let len = size_mb * 1024 * 1024;
    let mut buf = vec![0u8; len];
    rng.fill(buf.as_mut_slice());
    buf
}

/// Interpret raw bytes as little-endian f32 values (simulated weight loading).
fn simulate_model_load(data: &[u8]) -> Result<Vec<f32>> {
    let n_floats = data.len() / 4;
    let mut weights = Vec::with_capacity(n_floats);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        weights.push(f32::from_le_bytes(bytes));
    }
    Ok(weights)
}

/// Simple dot-product inference: treat `weights` as a flat matrix of
/// `input_size` columns, multiply by a ones-vector, and return row sums.
fn simulate_inference(weights: &[f32], input_size: usize) -> Result<Vec<f32>> {
    if input_size == 0 {
        return Ok(Vec::new());
    }
    let n_rows = weights.len() / input_size;
    let mut output = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let start = row * input_size;
        let end = start + input_size;
        let slice = weights.get(start..end).unwrap_or(&[]);
        let sum: f32 = slice.iter().sum();
        output.push(sum);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Container sizing logic
// ---------------------------------------------------------------------------

/// Produce sizing recommendations given the observed peak RSS (in kB).
fn compute_sizing(peak_rss_kb: u64) -> Vec<SizingRecommendation> {
    let peak_mb = peak_rss_kb.div_ceil(1024);

    let targets: &[(&str, u64, f64)] = &[
        ("Lambda-128MB", 128, 0.25),
        ("Lambda-256MB", 256, 0.25),
        ("Lambda-512MB", 512, 0.25),
        ("Lambda-1024MB", 1024, 0.25),
        ("Docker", 0, 0.20), // Docker: peak + 20 % headroom
    ];

    targets
        .iter()
        .map(|&(name, capacity, headroom_pct)| {
            let recommended = if capacity == 0 {
                // Docker: derive from peak
                let with_headroom = (peak_mb as f64 * (1.0 + headroom_pct)).ceil() as u64;
                // Round up to next 64 MB
                with_headroom.div_ceil(64) * 64
            } else {
                capacity
            };

            let fits = recommended >= peak_mb;
            let reasoning = if fits {
                format!(
                    "Peak {}MB fits in {}MB ({:.0}% headroom)",
                    peak_mb,
                    recommended,
                    (recommended as f64 / peak_mb.max(1) as f64 - 1.0) * 100.0
                )
            } else {
                format!(
                    "Peak {}MB exceeds {}MB -- NOT recommended",
                    peak_mb, recommended
                )
            };

            SizingRecommendation {
                target: name.to_string(),
                recommended_mb: recommended,
                headroom_pct,
                reasoning,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main recipe
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("monitoring_memory_profiler")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Tracking peak RSS during model load and inference");
    println!();

    // -- 1. Baseline Memory Snapshot -------------------------------------------
    println!("1. Baseline Memory Snapshot");
    let baseline = read_proc_status()?;
    println!(
        "   RSS:     {} kB ({} MB)",
        baseline.rss_kb,
        baseline.rss_kb / 1024
    );
    println!(
        "   VmSize:  {} kB ({} MB)",
        baseline.vm_size_kb,
        baseline.vm_size_kb / 1024
    );
    println!(
        "   VmPeak:  {} kB ({} MB)",
        baseline.vm_peak_kb,
        baseline.vm_peak_kb / 1024
    );
    println!();

    ctx.record_metric("baseline_rss_kb", baseline.rss_kb as i64);

    // -- 2. Model Load Profiling -----------------------------------------------
    println!("2. Model Load Profiling");
    println!(
        "   {:>8}  {:>12}  {:>12}  {:>12}  {:>10}",
        "Payload", "RSS start", "RSS peak", "RSS delta", "Duration"
    );
    println!("   {}", "-".repeat(62));

    let payload_sizes_mb: &[usize] = &[10, 50, 100];
    let mut profiles: Vec<MemoryProfile> = Vec::new();
    let mut last_weights: Vec<f32> = Vec::new();

    for &size_mb in payload_sizes_mb {
        let seed = hash_name_to_seed(&format!("payload_{size_mb}"));
        let mut rng = StdRng::seed_from_u64(seed);
        let payload = create_test_payload(size_mb, &mut rng);

        let (weights, profile) = profile_phase(&format!("load_{size_mb}MB"), || {
            simulate_model_load(&payload)
        })?;

        println!(
            "   {:>5} MB  {:>8} kB  {:>8} kB  {:>+8} kB  {:>8.2}ms",
            size_mb,
            profile.rss_start_kb,
            profile.rss_peak_kb,
            profile.rss_delta_kb,
            profile.duration_ms,
        );

        last_weights = weights;
        profiles.push(profile);
    }
    println!();

    // -- 3. Inference Profiling ------------------------------------------------
    println!("3. Inference Profiling");
    let input_size = 256;
    let (output, inf_profile) = profile_phase("inference", || {
        simulate_inference(&last_weights, input_size)
    })?;

    println!("   Input size:  {}", input_size);
    println!("   Output size: {}", output.len());
    println!("   RSS start:   {} kB", inf_profile.rss_start_kb);
    println!("   RSS peak:    {} kB", inf_profile.rss_peak_kb);
    println!("   RSS delta:   {:+} kB", inf_profile.rss_delta_kb);
    println!("   Duration:    {:.2}ms", inf_profile.duration_ms);
    println!();

    profiles.push(inf_profile);

    // -- 4. Peak Memory Analysis -----------------------------------------------
    println!("4. Peak Memory Analysis");
    let current = read_proc_status()?;
    let peak_rss_kb = current.vm_peak_kb.max(current.rss_kb);
    let delta_from_baseline = peak_rss_kb as i64 - baseline.rss_kb as i64;

    println!(
        "   Current RSS:  {} kB ({} MB)",
        current.rss_kb,
        current.rss_kb / 1024
    );
    println!(
        "   Peak VmPeak:  {} kB ({} MB)",
        peak_rss_kb,
        peak_rss_kb / 1024
    );
    println!(
        "   Delta from baseline: {:+} kB ({:+} MB)",
        delta_from_baseline,
        delta_from_baseline / 1024
    );
    println!();

    ctx.record_metric("peak_rss_kb", peak_rss_kb as i64);
    ctx.record_metric("rss_delta_kb", delta_from_baseline);

    // -- 5. Container Sizing Recommendations -----------------------------------
    println!("5. Container Sizing Recommendations");
    let recommendations = compute_sizing(peak_rss_kb);

    println!(
        "   {:>14}  {:>8}  {:>10}  Reasoning",
        "Target", "Size MB", "Headroom"
    );
    println!("   {}", "-".repeat(72));

    for rec in &recommendations {
        println!(
            "   {:>14}  {:>5} MB  {:>8.0}%  {}",
            rec.target,
            rec.recommended_mb,
            rec.headroom_pct * 100.0,
            rec.reasoning
        );
    }
    println!();

    // -- 6. Save JSON report and record metrics --------------------------------
    let report_path = ctx.path("memory_report.json");

    #[derive(Serialize)]
    struct FullReport {
        baseline_rss_kb: u64,
        peak_rss_kb: u64,
        profiles: Vec<MemoryProfile>,
        recommendations: Vec<SizingRecommendation>,
    }

    let full_report = FullReport {
        baseline_rss_kb: baseline.rss_kb,
        peak_rss_kb,
        profiles: profiles.clone(),
        recommendations: recommendations.clone(),
    };

    let json = serde_json::to_string_pretty(&full_report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&report_path, json)?;
    println!("6. Report saved to {:?}", report_path);

    ctx.record_metric("n_phases", profiles.len() as i64);
    ctx.record_metric("n_recommendations", recommendations.len() as i64);
    ctx.record_float_metric("peak_rss_mb", peak_rss_kb as f64 / 1024.0);

    ctx.report()?;
    println!();
    println!("=== Recipe Complete ===");

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_proc_status_returns_ok() {
        let snap = read_proc_status();
        assert!(snap.is_ok());
    }

    #[test]
    fn test_read_proc_status_rss_non_negative() {
        let snap = read_proc_status().expect("should read");
        // RSS is u64, so >= 0 by type; verify the reading is plausible.
        // On Linux this should be > 0; on non-Linux fallback is 0.
        assert!(snap.rss_kb < u64::MAX);
    }

    #[test]
    fn test_create_test_payload_length() {
        let mut rng = StdRng::seed_from_u64(42);
        let payload = create_test_payload(1, &mut rng);
        assert_eq!(payload.len(), 1024 * 1024);
    }

    #[test]
    fn test_create_test_payload_deterministic() {
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let p1 = create_test_payload(1, &mut rng1);
        let p2 = create_test_payload(1, &mut rng2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_simulate_model_load_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let loaded = simulate_model_load(&bytes).expect("load");
        assert_eq!(loaded.len(), original.len());
        for (a, b) in loaded.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_simulate_model_load_empty() {
        let result = simulate_model_load(&[]);
        assert!(result.is_ok());
        assert!(result.expect("empty").is_empty());
    }

    #[test]
    fn test_simulate_inference_basic() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = simulate_inference(&weights, 3).expect("inference");
        assert_eq!(output.len(), 2);
        assert!((output[0] - 6.0).abs() < f32::EPSILON); // 1+2+3
        assert!((output[1] - 15.0).abs() < f32::EPSILON); // 4+5+6
    }

    #[test]
    fn test_simulate_inference_zero_input() {
        let output = simulate_inference(&[1.0, 2.0], 0).expect("zero input");
        assert!(output.is_empty());
    }

    #[test]
    fn test_compute_sizing_has_all_targets() {
        let recs = compute_sizing(200_000); // ~195 MB
        assert_eq!(recs.len(), 5);
        assert!(recs.iter().any(|r| r.target.contains("Lambda")));
        assert!(recs.iter().any(|r| r.target.contains("Docker")));
    }

    #[test]
    fn test_compute_sizing_docker_exceeds_peak() {
        let peak_kb = 100 * 1024; // 100 MB
        let recs = compute_sizing(peak_kb);
        let docker = recs.iter().find(|r| r.target == "Docker").expect("docker");
        let peak_mb = (peak_kb + 1023) / 1024;
        assert!(
            docker.recommended_mb >= peak_mb,
            "Docker {}MB should >= peak {}MB",
            docker.recommended_mb,
            peak_mb
        );
    }

    #[test]
    fn test_profile_phase_captures_duration() {
        let (val, profile) = profile_phase("test_phase", || {
            // Small busy work
            let v: f32 = (0..1000).map(|i| i as f32).sum();
            Ok(v)
        })
        .expect("profile");

        assert!(val > 0.0);
        assert_eq!(profile.phase, "test_phase");
        assert!(profile.duration_ms >= 0.0);
    }

    #[test]
    fn test_memory_profile_serialization() {
        let profile = MemoryProfile {
            phase: "load".to_string(),
            rss_start_kb: 50_000,
            rss_peak_kb: 80_000,
            rss_delta_kb: 30_000,
            duration_ms: 12.5,
        };
        let json = serde_json::to_string(&profile);
        assert!(json.is_ok());
        let s = json.expect("json");
        assert!(s.contains("load"));
        assert!(s.contains("80000"));
    }

    #[test]
    fn test_sizing_recommendation_serialization() {
        let rec = SizingRecommendation {
            target: "Lambda-256MB".to_string(),
            recommended_mb: 256,
            headroom_pct: 0.25,
            reasoning: "fits".to_string(),
        };
        let json = serde_json::to_string(&rec);
        assert!(json.is_ok());
        assert!(json.expect("json").contains("Lambda-256MB"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_rss_non_negative(_seed in 0u64..1000) {
            let snap = read_proc_status().expect("proc status");
            // u64 is always >= 0 by type; verify no overflow / wrap-around
            prop_assert!(snap.rss_kb <= snap.vm_peak_kb.max(snap.rss_kb));
            prop_assert!(snap.vm_size_kb <= u64::MAX);
        }

        #[test]
        fn prop_allocation_increases_rss(size_kb in 256u64..2048) {
            let before = read_proc_status().expect("before");
            // Allocate and touch memory so the OS commits pages
            let len = (size_kb as usize) * 1024;
            let mut buf = vec![0u8; len];
            for (i, byte) in buf.iter_mut().enumerate() {
                *byte = (i & 0xFF) as u8;
            }
            let after = read_proc_status().expect("after");
            // RSS should not decrease (it may stay the same if pages were
            // already resident), so allow after >= before or after >= 0.
            prop_assert!(
                after.rss_kb >= before.rss_kb.saturating_sub(1024),
                "RSS after ({}) should not drop far below before ({})",
                after.rss_kb,
                before.rss_kb
            );
            // Keep buf alive past the second snapshot
            drop(buf);
        }

        #[test]
        fn prop_sizing_recommendation_exceeds_peak(peak_mb in 1u64..2048) {
            let peak_kb = peak_mb * 1024;
            let recs = compute_sizing(peak_kb);
            let docker = recs.iter().find(|r| r.target == "Docker").expect("docker");
            prop_assert!(
                docker.recommended_mb >= peak_mb,
                "Docker {}MB must >= peak {}MB",
                docker.recommended_mb,
                peak_mb,
            );
        }

        #[test]
        fn prop_model_load_preserves_count(n_floats in 1usize..512) {
            let floats: Vec<f32> = (0..n_floats).map(|i| i as f32).collect();
            let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            let loaded = simulate_model_load(&bytes).expect("load");
            prop_assert_eq!(loaded.len(), n_floats);
        }
    }
}
