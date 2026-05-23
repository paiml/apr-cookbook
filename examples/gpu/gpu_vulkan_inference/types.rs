//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::fmt;
use std::time::Instant;
use trueno::Matrix;

// ---------------------------------------------------------------------------
// GPU backend detection
// ---------------------------------------------------------------------------

/// Graphics API backend available on the current platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    None,
}

impl GpuBackend {
    /// Detect the preferred backend based on the target platform.
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::Vulkan
        }
        #[cfg(target_os = "macos")]
        {
            Self::Metal
        }
        #[cfg(target_os = "windows")]
        {
            Self::Dx12
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Self::None
        }
    }
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Dx12 => write!(f, "DirectX 12"),
            Self::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

/// Descriptor for a detected (or simulated) GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: GpuBackend,
    pub compute_units: u32,
    pub vram_mb: u32,
    pub vulkan_version: String,
}

impl GpuDeviceInfo {
    /// Peak theoretical GFLOPS for FP32 compute.
    pub fn peak_gflops(&self) -> f64 {
        // Intel Arc A770: 128 EUs * 8 ALUs * 2 (FMA) * 2.1 GHz ~ 4300 GFLOPS
        // Simulated: compute_units * 67 GFLOPS per EU (simplified)
        f64::from(self.compute_units) * 67.0
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] — {} CUs, {} MB VRAM, Vulkan {}",
            self.name, self.backend, self.compute_units, self.vram_mb, self.vulkan_version
        )
    }
}

// Try to detect an Intel Arc GPU by probing `/sys/class/drm`.
/// Falls back to a simulated Intel Arc A770 device.
pub fn detect_gpu_device() -> GpuDeviceInfo {
    // Attempt real detection on Linux via sysfs
    #[cfg(target_os = "linux")]
    {
        if let Some(info) = try_detect_from_sysfs() {
            return info;
        }
    }

    // Simulated Intel Arc A770 for demonstration
    GpuDeviceInfo {
        name: "Intel Arc A770 (Simulated)".to_string(),
        backend: GpuBackend::detect(),
        compute_units: 128,
        vram_mb: 16384,
        vulkan_version: "1.3.256".to_string(),
    }
}

/// Probe `/sys/class/drm/card*/device/vendor` for Intel (0x8086).
#[cfg(target_os = "linux")]
pub fn try_detect_from_sysfs() -> Option<GpuDeviceInfo> {
    use std::fs;
    let drm = std::path::Path::new("/sys/class/drm");
    if !drm.is_dir() {
        return None;
    }
    let entries = fs::read_dir(drm).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        if let Ok(vendor) = fs::read_to_string(&vendor_path) {
            if vendor.trim() == "0x8086" {
                return Some(GpuDeviceInfo {
                    name: format!("Intel iGPU ({})", name_str),
                    backend: GpuBackend::Vulkan,
                    compute_units: 64,
                    vram_mb: 4096,
                    vulkan_version: "1.3.0".to_string(),
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Vulkan pipeline simulation
// ---------------------------------------------------------------------------

/// Configuration for a simulated Vulkan compute pipeline.
#[derive(Debug, Clone)]
pub struct VulkanPipeline {
    pub workgroup_size: [u32; 3],
    pub buffer_size_bytes: usize,
    pub compute_shader: String,
}

impl VulkanPipeline {
    pub fn new(shader_name: &str) -> Self {
        Self {
            workgroup_size: [16, 16, 1],
            buffer_size_bytes: 256 * 1024 * 1024, // 256 MB default
            compute_shader: shader_name.to_string(),
        }
    }

    pub fn dispatch_groups(&self, m: usize, n: usize) -> [u32; 3] {
        let gx = (m as u32).div_ceil(self.workgroup_size[0]);
        let gy = (n as u32).div_ceil(self.workgroup_size[1]);
        [gx, gy, 1]
    }
}

impl fmt::Display for VulkanPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shader={}, workgroup=[{}x{}x{}], buffer={} MB",
            self.compute_shader,
            self.workgroup_size[0],
            self.workgroup_size[1],
            self.workgroup_size[2],
            self.buffer_size_bytes / (1024 * 1024)
        )
    }
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

pub fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
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
// Simulated Vulkan compute kernels
// ---------------------------------------------------------------------------

// Simulate a Vulkan compute-shader matrix multiply: C = A * B.
/// `a` is m x k, `b` is k x n, returns m x n.
pub fn simulate_vulkan_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// Simulate a Vulkan softmax kernel over `batch` sequences of `seq_len`.
pub fn simulate_vulkan_softmax(logits: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * seq_len];
    for b in 0..batch {
        let start = b * seq_len;
        let end = start + seq_len;
        let slice = &logits[start..end];

        let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for &v in slice {
            sum += (v - max_val).exp();
        }

        for (i, &v) in slice.iter().enumerate() {
            output[start + i] = (v - max_val).exp() / sum;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenchmarkResult {
    pub name: String,
    pub time_ms: f64,
    pub gflops: f64,
    pub backend: String,
}

/// Benchmark CPU matrix multiply via trueno for a given square size.
pub fn benchmark_cpu_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(size, size, 42);
    let b_data = generate_random_data(size, size, 43);

    let a = Matrix::from_vec(size, size, a_data).expect("matrix A creation failed");
    let b = Matrix::from_vec(size, size, b_data).expect("matrix B creation failed");

    // warmup
    let _ = a.matmul(&b);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = flops / (time_ms / 1000.0) / 1e9;

    BenchmarkResult {
        name: format!("cpu_matmul_{size}"),
        time_ms,
        gflops,
        backend: "CPU (trueno)".to_string(),
    }
}

/// Benchmark simulated Vulkan matrix multiply for a given square size.
pub fn benchmark_vulkan_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(size, size, 42);
    let b_data = generate_random_data(size, size, 43);

    // warmup
    let _ = simulate_vulkan_matmul(&a_data, &b_data, size, size, size);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simulate_vulkan_matmul(&a_data, &b_data, size, size, size);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = flops / (time_ms / 1000.0) / 1e9;

    BenchmarkResult {
        name: format!("vulkan_matmul_{size}"),
        time_ms,
        gflops,
        backend: "Vulkan (simulated)".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
