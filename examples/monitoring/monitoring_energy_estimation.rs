//! # Recipe: Inference Energy Estimation (RAPL)
//!
//! **Category**: Inference Monitoring
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: trueno (SIMD matrix ops)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A - Linux RAPL only)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Estimate per-inference energy consumption (joules) using Intel RAPL
//! (Running Average Power Limit). Gracefully falls back to TDP-based
//! estimation when RAPL sysfs is not readable.
//!
//! ## Toyota Way: ムダ (Muda) - Waste Elimination
//! Measure energy per inference to identify and eliminate computational waste.
//!
//! ## Run Command
//! ```bash
//! cargo run --example monitoring_energy_estimation
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use trueno::Matrix;

// ---------------------------------------------------------------------------
// RAPL reader
// ---------------------------------------------------------------------------

/// Reads energy counters from the Intel RAPL sysfs interface.
struct RaplReader {
    path: PathBuf,
    available: bool,
}

impl RaplReader {
    /// Probe `/sys/class/powercap/intel-rapl:0/energy_uj` for readability.
    fn new() -> Self {
        let path = PathBuf::from("/sys/class/powercap/intel-rapl:0/energy_uj");
        let available = std::fs::read_to_string(&path).is_ok();
        Self { path, available }
    }

    /// Read the current energy counter in micro-joules.
    /// Returns `Ok(None)` when RAPL is not available.
    fn read_energy_uj(&self) -> Result<Option<u64>> {
        if !self.available {
            return Ok(None);
        }
        let Ok(content) = std::fs::read_to_string(&self.path) else {
            return Ok(None);
        };
        let value: u64 = content
            .trim()
            .parse()
            .map_err(|e: std::num::ParseIntError| CookbookError::Aprender(e.to_string()))?;
        Ok(Some(value))
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ---------------------------------------------------------------------------
// Energy data types
// ---------------------------------------------------------------------------

/// A single energy measurement across a batch of inferences.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EnergyMeasurement {
    energy_uj: u64,
    duration: Duration,
    joules: f64,
    watts: f64,
}

/// Serializable report summarising energy usage.
#[derive(Debug, Clone, Serialize)]
struct InferenceEnergyReport {
    model_name: String,
    n_inferences: usize,
    total_joules: f64,
    joules_per_inference: f64,
    watts_avg: f64,
    co2_grams: f64,
    method: String,
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash = hasher.finish();
        data.push((hash as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    data
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Fallback energy estimate from TDP rating and CPU time.
///
/// `utilization_factor` accounts for the fact that the workload rarely
/// saturates all cores at the rated TDP.
fn estimate_from_tdp(duration: Duration, tdp_watts: f64) -> f64 {
    const UTILIZATION_FACTOR: f64 = 0.35;
    tdp_watts * duration.as_secs_f64() * UTILIZATION_FACTOR
}

/// Measure energy consumed by `workload` over `iterations` invocations.
fn measure_inference_energy(
    rapl: &RaplReader,
    workload: impl Fn(),
    iterations: usize,
) -> Result<EnergyMeasurement> {
    let before_uj = rapl.read_energy_uj()?;

    let start = Instant::now();
    for _ in 0..iterations {
        workload();
    }
    let duration = start.elapsed();

    let after_uj = rapl.read_energy_uj()?;

    let (energy_uj, joules) = if let (Some(b), Some(a)) = (before_uj, after_uj) {
        let delta = if a >= b { a - b } else { a }; // counter wrap
        (delta, delta as f64 / 1_000_000.0)
    } else {
        // TDP fallback for Intel Core Ultra 7 155H (28 W PBP)
        let est = estimate_from_tdp(duration, 28.0);
        let uj = (est * 1_000_000.0) as u64;
        (uj, est)
    };

    let secs = duration.as_secs_f64();
    let watts = if secs > 0.0 { joules / secs } else { 0.0 };

    Ok(EnergyMeasurement {
        energy_uj,
        duration,
        joules,
        watts,
    })
}

/// Return a closure that performs a matrix multiply of size `size x size`.
fn simulate_inference_workload(size: usize) -> impl Fn() {
    let a_data = generate_random_data(size, size, 42);
    let b_data = generate_random_data(size, size, 43);
    let a = Matrix::from_vec(size, size, a_data).expect("matrix A creation");
    let b = Matrix::from_vec(size, size, b_data).expect("matrix B creation");
    move || {
        let _ = a.matmul(&b);
    }
}

/// Convert joules to grams of CO2 using the US grid average.
///
/// US EPA 2024 average: 0.417 g CO2 per Wh (0.000_417 kg CO2/Wh).
fn joules_to_co2_grams(joules: f64) -> f64 {
    let watt_hours = joules / 3600.0;
    watt_hours * 0.417
}

// ---------------------------------------------------------------------------
// Main recipe
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("monitoring_energy_estimation")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Estimating per-inference energy consumption via RAPL / TDP fallback");
    println!();

    // ── 1. RAPL Detection ──────────────────────────────────────────────
    println!("1. RAPL Detection");
    let rapl = RaplReader::new();
    println!("   Path:      {:?}", rapl.path);
    println!("   Available: {}", rapl.is_available());

    if let Some(initial_uj) = rapl.read_energy_uj()? {
        println!("   Initial counter: {} uJ", initial_uj);
    } else {
        println!("   RAPL not readable -- will use TDP estimation (28 W)");
    }
    println!();

    let method = if rapl.is_available() {
        "rapl"
    } else {
        "estimated"
    };

    // ── 2. Workload Configuration ──────────────────────────────────────
    println!("2. Workload Configuration");
    let sizes: &[usize] = &[128, 256, 512];
    let iterations = 20;
    println!("   Matrix sizes: {:?}", sizes);
    println!("   Iterations per size: {}", iterations);
    println!();

    // ── 3. Energy Measurement ──────────────────────────────────────────
    println!("3. Energy Measurement");
    println!(
        "   {:>6}  {:>12}  {:>10}  {:>14}",
        "Size", "Joules", "Watts", "J/inference"
    );
    println!("   {}", "-".repeat(50));

    let mut reports: Vec<InferenceEnergyReport> = Vec::new();

    for &size in sizes {
        let workload = simulate_inference_workload(size);
        let measurement = measure_inference_energy(&rapl, workload, iterations)?;

        let j_per_inf = measurement.joules / iterations as f64;
        let co2 = joules_to_co2_grams(measurement.joules);

        println!(
            "   {:>6}  {:>12.6}  {:>10.3}  {:>14.8}",
            size, measurement.joules, measurement.watts, j_per_inf
        );

        reports.push(InferenceEnergyReport {
            model_name: format!("matmul_{}x{}", size, size),
            n_inferences: iterations,
            total_joules: measurement.joules,
            joules_per_inference: j_per_inf,
            watts_avg: measurement.watts,
            co2_grams: co2,
            method: method.to_string(),
        });
    }
    println!();

    // ── 4. CO2 Estimation ──────────────────────────────────────────────
    println!("4. CO2 Estimation (US grid avg 0.417 g/Wh)");
    for report in &reports {
        println!(
            "   {}: {:.6} g CO2 ({} inferences)",
            report.model_name, report.co2_grams, report.n_inferences
        );
    }
    println!();

    // ── 5. Efficiency Analysis ─────────────────────────────────────────
    println!("5. Efficiency Analysis");
    println!(
        "   {:>18}  {:>14}  {:>10}",
        "Model", "J/inference", "CO2 (g)"
    );
    println!("   {}", "-".repeat(48));
    for report in &reports {
        println!(
            "   {:>18}  {:>14.8}  {:>10.6}",
            report.model_name,
            report.joules_per_inference,
            report.co2_grams / report.n_inferences as f64
        );
    }
    println!();

    // ── 6. Save & Record ───────────────────────────────────────────────
    let report_path = ctx.path("energy_report.json");
    let json = serde_json::to_string_pretty(&reports)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&report_path, json)?;
    println!("6. Report saved to {:?}", report_path);

    if let Some(last) = reports.last() {
        ctx.record_float_metric("joules_per_inference", last.joules_per_inference);
        ctx.record_float_metric("watts_avg", last.watts_avg);
        ctx.record_float_metric("co2_grams_total", last.co2_grams);
        ctx.record_string_metric("method", &last.method);
        ctx.record_metric("n_workloads", reports.len() as i64);
    }

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
    fn test_rapl_reader_creation() {
        let rapl = RaplReader::new();
        // Must not panic; availability depends on host.
        assert!(!rapl.path.as_os_str().is_empty());
    }

    #[test]
    fn test_rapl_read_returns_ok() {
        let rapl = RaplReader::new();
        let result = rapl.read_energy_uj();
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimate_from_tdp_positive() {
        let dur = Duration::from_millis(100);
        let joules = estimate_from_tdp(dur, 28.0);
        assert!(joules > 0.0);
    }

    #[test]
    fn test_estimate_from_tdp_zero_duration() {
        let joules = estimate_from_tdp(Duration::ZERO, 28.0);
        assert!((joules - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_from_tdp_scales_with_watts() {
        let dur = Duration::from_secs(1);
        let j_low = estimate_from_tdp(dur, 15.0);
        let j_high = estimate_from_tdp(dur, 65.0);
        assert!(j_high > j_low);
    }

    #[test]
    fn test_joules_to_co2_zero() {
        let co2 = joules_to_co2_grams(0.0);
        assert!((co2 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_joules_to_co2_positive() {
        let co2 = joules_to_co2_grams(3600.0); // 1 Wh
        assert!((co2 - 0.417).abs() < 1e-6);
    }

    #[test]
    fn test_joules_to_co2_proportional() {
        let co2_a = joules_to_co2_grams(100.0);
        let co2_b = joules_to_co2_grams(200.0);
        assert!((co2_b - 2.0 * co2_a).abs() < 1e-9);
    }

    #[test]
    fn test_generate_random_data_deterministic() {
        let d1 = generate_random_data(8, 8, 42);
        let d2 = generate_random_data(8, 8, 42);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_generate_random_data_length() {
        let d = generate_random_data(4, 6, 1);
        assert_eq!(d.len(), 24);
    }

    #[test]
    fn test_simulate_workload_runs() {
        let workload = simulate_inference_workload(32);
        workload(); // must not panic
    }

    #[test]
    fn test_measure_inference_energy_ok() {
        let rapl = RaplReader::new();
        let workload = simulate_inference_workload(32);
        let result = measure_inference_energy(&rapl, workload, 2);
        assert!(result.is_ok());
        let m = result.expect("measurement");
        assert!(m.joules >= 0.0);
        assert!(m.duration > Duration::ZERO);
    }

    #[test]
    fn test_energy_report_serialization() {
        let report = InferenceEnergyReport {
            model_name: "test".to_string(),
            n_inferences: 10,
            total_joules: 0.5,
            joules_per_inference: 0.05,
            watts_avg: 5.0,
            co2_grams: 0.001,
            method: "estimated".to_string(),
        };
        let json = serde_json::to_string(&report);
        assert!(json.is_ok());
        let s = json.expect("json");
        assert!(s.contains("test"));
        assert!(s.contains("estimated"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_energy_non_negative(
            dur_ms in 0u64..10_000,
            tdp in 1.0f64..200.0
        ) {
            let duration = Duration::from_millis(dur_ms);
            let joules = estimate_from_tdp(duration, tdp);
            prop_assert!(joules >= 0.0, "energy must be non-negative, got {}", joules);
        }

        #[test]
        fn prop_co2_proportional_to_energy(
            joules in 0.0f64..1_000_000.0
        ) {
            let co2 = joules_to_co2_grams(joules);
            prop_assert!(co2 >= 0.0, "CO2 must be non-negative");
            if joules > 0.0 {
                let ratio = co2 / joules;
                // ratio should be constant: 0.417 / 3600
                let expected_ratio = 0.417 / 3600.0;
                prop_assert!(
                    (ratio - expected_ratio).abs() < 1e-12,
                    "ratio {} differs from expected {}", ratio, expected_ratio
                );
            }
        }

        #[test]
        fn prop_estimation_bounded(
            tdp in 5.0f64..250.0
        ) {
            let duration = Duration::from_secs(1);
            let joules = estimate_from_tdp(duration, tdp);
            // utilization factor is 0.35, so joules = tdp * 0.35
            let expected = tdp * 0.35;
            prop_assert!(
                (joules - expected).abs() < 1e-9,
                "got {} expected {}", joules, expected
            );
            // sanity: bounded by TDP itself
            prop_assert!(joules <= tdp, "joules {} > tdp {}", joules, tdp);
        }

        #[test]
        fn prop_random_data_bounded(
            seed in 0u64..10_000
        ) {
            let data = generate_random_data(4, 4, seed);
            for &v in &data {
                prop_assert!(v >= -1.0 && v <= 1.0,
                    "value {} out of [-1, 1]", v);
            }
        }
    }
}
