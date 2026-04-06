#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
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
pub struct MemorySnapshot {
    pub rss_kb: u64,
    pub vm_size_kb: u64,
    pub vm_peak_kb: u64,
    pub timestamp: Instant,
}

/// Serializable profile of a single phase (load, inference, etc.).
#[derive(Debug, Clone, Serialize)]
pub struct MemoryProfile {
    pub phase: String,
    pub rss_start_kb: u64,
    pub rss_peak_kb: u64,
    pub rss_delta_kb: i64,
    pub duration_ms: f64,
}

/// Serializable container sizing recommendation.
#[derive(Debug, Clone, Serialize)]
pub struct SizingRecommendation {
    pub target: String,
    pub recommended_mb: u64,
    pub headroom_pct: f64,
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// /proc/self/status parser
// ---------------------------------------------------------------------------

// Read memory counters from `/proc/self/status`.
//
// Falls back to zeros with a printed warning when the file is unreadable
/// (e.g., macOS, WASM, restricted containers).
pub fn read_proc_status() -> Result<MemorySnapshot> {
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

// Run `f`, bracketed by memory snapshots, and return the result together with
/// a [`MemoryProfile`] capturing the RSS delta.
pub fn profile_phase<F, T>(name: &str, f: F) -> Result<(T, MemoryProfile)>
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
pub fn create_test_payload(size_mb: usize, rng: &mut StdRng) -> Vec<u8> {
    let len = size_mb * 1024 * 1024;
    let mut buf = vec![0u8; len];
    rng.fill(buf.as_mut_slice());
    buf
}

/// Interpret raw bytes as little-endian f32 values (simulated weight loading).
pub fn simulate_model_load(data: &[u8]) -> Result<Vec<f32>> {
    let n_floats = data.len() / 4;
    let mut weights = Vec::with_capacity(n_floats);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        weights.push(f32::from_le_bytes(bytes));
    }
    Ok(weights)
}

// Simple dot-product inference: treat `weights` as a flat matrix of
/// `input_size` columns, multiply by a ones-vector, and return row sums.
pub fn simulate_inference(weights: &[f32], input_size: usize) -> Result<Vec<f32>> {
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
pub fn compute_sizing(peak_rss_kb: u64) -> Vec<SizingRecommendation> {
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
