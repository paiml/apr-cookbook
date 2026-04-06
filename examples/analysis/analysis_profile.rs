//! # APR Model Profiling (Roofline Analysis)
//!
//! CLI equivalent: `apr profile model.apr --granular`
//!
//! Performs roofline model analysis to classify each layer as compute-bound
//! or memory-bound. Produces per-layer profiling, an ASCII roofline chart,
//! bottleneck identification, and optimization recommendations.
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    Compute,
    Memory,
}

impl std::fmt::Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bound::Compute => write!(f, "COMPUTE"),
            Bound::Memory => write!(f, "MEMORY"),
        }
    }
}

#[derive(Debug, Clone)]
struct ProfileResult {
    layer_name: String,
    flops: u64,
    bytes_accessed: u64,
    arithmetic_intensity: f64,
    bound: Bound,
}

#[derive(Debug, Clone)]
struct HardwareSpec {
    peak_gflops: f64,
    memory_bandwidth_gb_s: f64,
    name: String,
}

impl HardwareSpec {
    fn ridge_point(&self) -> f64 {
        // Arithmetic intensity at which compute and memory ceilings meet
        self.peak_gflops / self.memory_bandwidth_gb_s
    }
}

#[derive(Debug, Clone)]
struct Recommendation {
    layer: String,
    bound: Bound,
    suggestion: String,
    priority: u8, // 1 = high, 3 = low
}

// ---------------------------------------------------------------------------
// Analysis logic
// ---------------------------------------------------------------------------

fn roofline_analysis(flops: u64, bytes_accessed: u64, hw: &HardwareSpec) -> (f64, Bound) {
    let arithmetic_intensity = if bytes_accessed > 0 {
        flops as f64 / bytes_accessed as f64
    } else {
        f64::MAX
    };

    let ridge = hw.ridge_point();
    let bound = if arithmetic_intensity < ridge {
        Bound::Memory
    } else {
        Bound::Compute
    };

    (arithmetic_intensity, bound)
}

fn estimate_layer_profile(
    name: &str,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    hw: &HardwareSpec,
) -> ProfileResult {
    // FLOPs for dense matmul: 2 * M * N * K
    let flops = 2 * batch_size as u64 * output_dim as u64 * input_dim as u64;

    // Bytes accessed: read weights + read input + write output
    let weight_bytes = (input_dim * output_dim * 4) as u64;
    let input_bytes = (batch_size * input_dim * 4) as u64;
    let output_bytes = (batch_size * output_dim * 4) as u64;
    let bytes_accessed = weight_bytes + input_bytes + output_bytes;

    let (arithmetic_intensity, bound) = roofline_analysis(flops, bytes_accessed, hw);

    ProfileResult {
        layer_name: name.to_string(),
        flops,
        bytes_accessed,
        arithmetic_intensity,
        bound,
    }
}

fn generate_recommendations(profiles: &[ProfileResult], hw: &HardwareSpec) -> Vec<Recommendation> {
    let mut recs = Vec::new();

    for p in profiles {
        match p.bound {
            Bound::Memory => {
                recs.push(Recommendation {
                    layer: p.layer_name.clone(),
                    bound: Bound::Memory,
                    suggestion: "Quantize weights (FP32 -> INT8) to reduce memory traffic"
                        .to_string(),
                    priority: 1,
                });
                if p.arithmetic_intensity < hw.ridge_point() * 0.1 {
                    recs.push(Recommendation {
                        layer: p.layer_name.clone(),
                        bound: Bound::Memory,
                        suggestion: "Consider weight pruning to reduce tensor size".to_string(),
                        priority: 2,
                    });
                }
            }
            Bound::Compute => {
                recs.push(Recommendation {
                    layer: p.layer_name.clone(),
                    bound: Bound::Compute,
                    suggestion: "Use SIMD/GPU acceleration for compute-bound layers".to_string(),
                    priority: 2,
                });
                if p.flops > 1_000_000_000 {
                    recs.push(Recommendation {
                        layer: p.layer_name.clone(),
                        bound: Bound::Compute,
                        suggestion: "Consider knowledge distillation to reduce model complexity"
                            .to_string(),
                        priority: 3,
                    });
                }
            }
        }
    }

    recs.sort_by_key(|r| r.priority);
    recs
}

fn render_roofline_ascii(profiles: &[ProfileResult], hw: &HardwareSpec) -> String {
    let width = 60;
    let height = 15;
    let mut grid = vec![vec![' '; width]; height];

    // Determine axis ranges
    let max_ai = profiles
        .iter()
        .map(|p| p.arithmetic_intensity)
        .fold(0.0_f64, f64::max)
        .max(hw.ridge_point() * 2.0);

    let peak = hw.peak_gflops;

    // Draw memory roof (diagonal line from origin to ridge point)
    let ridge = hw.ridge_point();
    #[allow(clippy::needless_range_loop)]
    for x in 0..width {
        let ai = (x as f64 / width as f64) * max_ai;
        let perf = ai * hw.memory_bandwidth_gb_s;
        let perf_clamped = perf.min(peak);
        let y = ((perf_clamped / peak) * (height - 2) as f64) as usize;
        let y = y.min(height - 2);
        let row = height - 2 - y;
        if row < height {
            grid[row][x] = if ai <= ridge { '/' } else { '-' };
        }
    }

    // Plot layer points
    let symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
    for (i, p) in profiles.iter().enumerate() {
        let x = ((p.arithmetic_intensity / max_ai) * (width - 1) as f64) as usize;
        let x = x.min(width - 1);
        let perf = p.arithmetic_intensity * hw.memory_bandwidth_gb_s;
        let perf_clamped = perf.min(peak);
        let y = ((perf_clamped / peak) * (height - 2) as f64) as usize;
        let y = y.min(height - 2);
        let row = height - 2 - y;
        let sym = symbols[i % symbols.len()];
        if row < height && x < width {
            grid[row][x] = sym;
        }
    }

    let mut output = String::new();
    output.push_str(&format!(
        "  Roofline: {:.0} GFLOP/s peak, {:.0} GB/s bandwidth\n",
        hw.peak_gflops, hw.memory_bandwidth_gb_s
    ));
    output.push_str(&format!("  Ridge point: {:.2} FLOP/B\n\n", ridge));
    output.push_str("  GFLOP/s\n");
    for (i, row) in grid.iter().enumerate() {
        let perf_val = peak * (1.0 - i as f64 / (height - 1) as f64);
        let line: String = row.iter().collect();
        output.push_str(&format!("  {:>6.0} |{line}\n", perf_val));
    }
    output.push_str(&format!("         +{}\n", "-".repeat(width)));
    output.push_str("          Arithmetic Intensity (FLOP/B)\n\n");

    // Legend
    for (i, p) in profiles.iter().enumerate() {
        let sym = symbols[i % symbols.len()];
        output.push_str(&format!(
            "  {sym} = {} (AI={:.2}, {})\n",
            p.layer_name, p.arithmetic_intensity, p.bound
        ));
    }

    output
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_profile")?;

    println!("=== APR Model Profiler (Roofline Analysis) ===\n");

    // --- Section 1: Define hardware target ---
    let hw = HardwareSpec {
        peak_gflops: 100.0,          // e.g., mid-range CPU
        memory_bandwidth_gb_s: 50.0, // DDR4 bandwidth
        name: "Intel i7-12700 (DDR4-3200)".to_string(),
    };
    println!("Hardware: {}", hw.name);
    println!("Peak compute:     {:.0} GFLOP/s", hw.peak_gflops);
    println!("Memory bandwidth: {:.0} GB/s", hw.memory_bandwidth_gb_s);
    println!("Ridge point:      {:.2} FLOP/B\n", hw.ridge_point());

    // --- Section 2: Create and profile model layers ---
    println!("--- Per-Layer Profiling ---");

    let batch_size = 32;
    let layers = vec![
        ("embedding", 1000, 128),
        ("attention.qkv", 128, 384),
        ("attention.out", 384, 128),
        ("ffn.up", 128, 512),
        ("ffn.down", 512, 128),
        ("output.proj", 128, 1000),
    ];

    let mut profiles = Vec::new();
    for (name, in_dim, out_dim) in &layers {
        let p = estimate_layer_profile(name, *in_dim, *out_dim, batch_size, &hw);
        profiles.push(p);
    }

    // Create a model bundle for context
    let seed = hash_name_to_seed("profile-model");
    let payload = generate_model_payload(seed, 128 * 128);
    let bundle = ModelBundleV2::new()
        .with_name("profile-target")
        .with_description("Model for roofline profiling")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![128, 128], payload)
        .build();
    std::fs::write(ctx.path("profile-target.apr"), &bundle)?;

    println!(
        "\n{:<18} {:>12} {:>12} {:>8} {:>10}",
        "Layer", "FLOP", "Bytes", "AI", "Bound"
    );
    println!("{}", "-".repeat(65));
    for p in &profiles {
        println!(
            "{:<18} {:>12} {:>12} {:>8.2} {:>10}",
            p.layer_name, p.flops, p.bytes_accessed, p.arithmetic_intensity, p.bound
        );
    }

    // --- Section 3: Roofline chart ---
    println!("\n--- Roofline Chart ---\n");
    let chart = render_roofline_ascii(&profiles, &hw);
    println!("{chart}");

    // --- Section 4: Bottleneck identification ---
    println!("--- Bottleneck Identification ---");
    let memory_bound: Vec<_> = profiles
        .iter()
        .filter(|p| p.bound == Bound::Memory)
        .collect();
    let compute_bound: Vec<_> = profiles
        .iter()
        .filter(|p| p.bound == Bound::Compute)
        .collect();

    println!("Memory-bound layers ({}):", memory_bound.len());
    for p in &memory_bound {
        println!(
            "  - {} (AI={:.2}, {:.0} bytes accessed)",
            p.layer_name, p.arithmetic_intensity, p.bytes_accessed
        );
    }
    println!("Compute-bound layers ({}):", compute_bound.len());
    for p in &compute_bound {
        println!(
            "  - {} (AI={:.2}, {:.0} FLOP)",
            p.layer_name, p.arithmetic_intensity, p.flops
        );
    }

    // --- Section 5: Optimization recommendations ---
    println!("\n--- Optimization Recommendations ---");
    let recs = generate_recommendations(&profiles, &hw);
    assert!(!recs.is_empty(), "Recommendations must not be empty");

    for (i, rec) in recs.iter().enumerate() {
        let priority_label = match rec.priority {
            1 => "HIGH",
            2 => "MEDIUM",
            _ => "LOW",
        };
        println!(
            "  {}. [{}] {} ({}): {}",
            i + 1,
            priority_label,
            rec.layer,
            rec.bound,
            rec.suggestion,
        );
    }

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hw() -> HardwareSpec {
        HardwareSpec {
            peak_gflops: 100.0,
            memory_bandwidth_gb_s: 50.0,
            name: "test-cpu".to_string(),
        }
    }

    #[test]
    fn test_ridge_point_calculation() {
        let hw = test_hw();
        let ridge = hw.ridge_point();
        assert!((ridge - 2.0).abs() < 1e-6, "100/50 = 2.0 FLOP/B");
    }

    #[test]
    fn test_memory_bound_classification() {
        let hw = test_hw();
        // Low arithmetic intensity -> memory bound
        let (ai, bound) = roofline_analysis(100, 1000, &hw);
        assert_eq!(bound, Bound::Memory);
        assert!(ai < hw.ridge_point());
    }

    #[test]
    fn test_compute_bound_classification() {
        let hw = test_hw();
        // High arithmetic intensity -> compute bound
        let (ai, bound) = roofline_analysis(100_000, 100, &hw);
        assert_eq!(bound, Bound::Compute);
        assert!(ai >= hw.ridge_point());
    }

    #[test]
    fn test_arithmetic_intensity_correct() {
        let hw = test_hw();
        let (ai, _) = roofline_analysis(200, 100, &hw);
        assert!((ai - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_arithmetic_intensity_zero_bytes() {
        let hw = test_hw();
        let (ai, bound) = roofline_analysis(100, 0, &hw);
        assert_eq!(ai, f64::MAX);
        assert_eq!(bound, Bound::Compute);
    }

    #[test]
    fn test_layer_profile_flops() {
        let hw = test_hw();
        let p = estimate_layer_profile("test", 64, 32, 8, &hw);
        // 2 * batch * out * in = 2 * 8 * 32 * 64 = 32768
        assert_eq!(p.flops, 32768);
    }

    #[test]
    fn test_layer_profile_bytes() {
        let hw = test_hw();
        let p = estimate_layer_profile("test", 64, 32, 8, &hw);
        // weights: 64*32*4 = 8192, input: 8*64*4 = 2048, output: 8*32*4 = 1024
        assert_eq!(p.bytes_accessed, 8192 + 2048 + 1024);
    }

    #[test]
    fn test_recommendations_nonempty() {
        let hw = test_hw();
        let profiles = vec![
            estimate_layer_profile("embed", 1000, 128, 1, &hw),
            estimate_layer_profile("ffn", 128, 512, 32, &hw),
        ];
        let recs = generate_recommendations(&profiles, &hw);
        assert!(!recs.is_empty());
    }

    #[test]
    fn test_recommendations_sorted_by_priority() {
        let hw = test_hw();
        let profiles = vec![
            estimate_layer_profile("a", 1000, 128, 1, &hw),
            estimate_layer_profile("b", 128, 512, 32, &hw),
        ];
        let recs = generate_recommendations(&profiles, &hw);
        for i in 1..recs.len() {
            assert!(recs[i].priority >= recs[i - 1].priority);
        }
    }

    #[test]
    fn test_roofline_chart_renders() {
        let hw = test_hw();
        let profiles = vec![
            estimate_layer_profile("layer_a", 64, 32, 8, &hw),
            estimate_layer_profile("layer_b", 128, 256, 32, &hw),
        ];
        let chart = render_roofline_ascii(&profiles, &hw);
        assert!(chart.contains("Roofline"));
        assert!(chart.contains("layer_a"));
        assert!(chart.contains("layer_b"));
    }

    #[test]
    fn test_bound_display() {
        assert_eq!(format!("{}", Bound::Compute), "COMPUTE");
        assert_eq!(format!("{}", Bound::Memory), "MEMORY");
    }

    #[test]
    fn test_hardware_different_specs() {
        let gpu = HardwareSpec {
            peak_gflops: 10000.0,
            memory_bandwidth_gb_s: 900.0,
            name: "A100".to_string(),
        };
        let ridge = gpu.ridge_point();
        assert!(ridge > 10.0, "GPU ridge point should be higher");
    }
}
