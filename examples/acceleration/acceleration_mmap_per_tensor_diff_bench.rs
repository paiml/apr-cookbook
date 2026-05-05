//! # Acceleration — APR mmap per-tensor diff bench
//!
//! aprender PR #1058 added `memmap2`-backed lazy tensor load to
//! `load_tensor_f32`, unblocking `apr diff --values` on 7B-parameter models
//! (12+ min → 192s for full 339-tensor sweep). This recipe demonstrates
//! the pattern: write N synthetic tensor blobs to disk, then mmap each,
//! diff against an in-memory reference, and report timing.
//!
//! Demonstrates the **ACC+.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PR #1058 + Linux mmap(2) + Bonwick (1994). The Slab Allocator. USENIX Summer.
//!
//! Run with: cargo run --example acceleration_mmap_per_tensor_diff_bench
//!
//! Added by PMAT-085 (expand-cookbooks: Tier 3 perf benches).

#![allow(unsafe_code)] // Mmap::map is the canonical entry-point; same pattern as F2 mmap falsification

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn write_tensor(path: &std::path::Path, data: &[f32]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    for x in data {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn mmap_tensor_max_abs_error(path: &std::path::Path, reference: &[f32]) -> std::io::Result<f64> {
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    let n = reference.len();
    assert_eq!(mmap.len(), n * 4, "mmap length must match expected tensor");
    let mut max_abs = 0.0f64;
    for i in 0..n {
        let bytes: [u8; 4] = mmap[i * 4..i * 4 + 4].try_into().unwrap();
        let v = f32::from_le_bytes(bytes);
        let d = (v as f64 - reference[i] as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    Ok(max_abs)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("acceleration_mmap_per_tensor_diff_bench")?;
    let dir = tempfile::tempdir()?;

    // Write 32 synthetic tensors of 4096 f32 each.
    let n_tensors = 32usize;
    let dim = 4096usize;
    let reference: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();

    let t0 = Instant::now();
    for i in 0..n_tensors {
        let path = dir.path().join(format!("t{i:04}.f32"));
        write_tensor(&path, &reference)?;
    }
    let write_ns = t0.elapsed().as_nanos();

    let t1 = Instant::now();
    let mut total_max_abs = 0.0f64;
    for i in 0..n_tensors {
        let path = dir.path().join(format!("t{i:04}.f32"));
        let m = mmap_tensor_max_abs_error(&path, &reference)?;
        total_max_abs = total_max_abs.max(m);
    }
    let diff_ns = t1.elapsed().as_nanos();

    println!("mmap-backed per-tensor diff bench ({n_tensors} tensors × {dim} f32):");
    println!("  write ns:    {:>10}", write_ns);
    println!("  diff ns:     {:>10}", diff_ns);
    println!("  max abs err: {total_max_abs:.6} (should be 0 for identical tensors)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tensors_yield_zero_max_abs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.f32");
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        write_tensor(&path, &data).unwrap();
        let m = mmap_tensor_max_abs_error(&path, &data).unwrap();
        assert_eq!(m, 0.0);
    }

    #[test]
    fn perturbed_tensor_yields_positive_max_abs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.f32");
        let written: Vec<f32> = (0..16).map(|i| i as f32 + 0.001).collect();
        let reference: Vec<f32> = (0..16).map(|i| i as f32).collect();
        write_tensor(&path, &written).unwrap();
        let m = mmap_tensor_max_abs_error(&path, &reference).unwrap();
        assert!(m > 0.0);
    }
}
