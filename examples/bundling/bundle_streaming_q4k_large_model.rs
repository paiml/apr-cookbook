//! # Bundling — Streaming APR→Q4K for ≥4 GiB Models (ALB-093)
//!
//! aprender ALB-093/GH-434 added a streaming quantize path that doesn't
//! OOM on ≥4 GiB models. This recipe demonstrates the streaming pattern:
//! synthesize tensors in chunks, quantize chunk-by-chunk, write incrementally.
//! Uses small chunks for IIUR offline-only (the real ≥4 GiB scenario can't
//! run in CI, but the streaming arithmetic is identical at every scale).
//!
//! Demonstrates the **BND+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ALB-093 / GH-434 + Frantar et al. (2023). GPTQ. arXiv:2210.17323
//!
//! Run with: cargo run --example bundle_streaming_q4k_large_model
//!
//! Added by PMAT-085 (expand-cookbooks: Tier 3 perf benches).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::fs::File;
use std::io::Write;

const TOTAL_PARAMS: usize = 1_000_000; // 1M synthetic; real path scales to ≥4 GiB
const CHUNK_SIZE: usize = 65_536; // 64K f32 elements per chunk

/// Faux Q4K quantization: pack 8 f32 values into 4 bytes (1 byte per pair) +
/// per-group scale. Real Q4K is more involved; this is a structural stand-in
/// to demonstrate the chunked streaming pattern.
fn quantize_chunk_q4k(chunk: &[f32]) -> Vec<u8> {
    let scale: f32 = chunk
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()))
        .max(1e-9);
    let mut out = Vec::with_capacity(4 + chunk.len() / 2);
    out.extend(scale.to_le_bytes());
    for pair in chunk.chunks(2) {
        let a = ((pair[0] / scale * 7.0).clamp(-8.0, 7.0) as i8) & 0x0F;
        let b = if pair.len() == 2 {
            ((pair[1] / scale * 7.0).clamp(-8.0, 7.0) as i8) & 0x0F
        } else {
            0
        };
        out.push((a as u8) | ((b as u8) << 4));
    }
    out
}

fn streaming_quantize_to_disk(out_path: &std::path::Path) -> std::io::Result<usize> {
    let mut file = File::create(out_path)?;
    let mut total_in_bytes = 0;
    let mut total_out_bytes = 0;
    for chunk_idx in 0..(TOTAL_PARAMS.div_ceil(CHUNK_SIZE)) {
        let start = chunk_idx * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(TOTAL_PARAMS);
        let chunk: Vec<f32> = (start..end).map(|i| (i as f32) * 0.0001).collect();
        total_in_bytes += chunk.len() * 4;
        let q = quantize_chunk_q4k(&chunk);
        total_out_bytes += q.len();
        file.write_all(&q)?;
    }
    drop(file);
    println!("streaming quantize: in={total_in_bytes} bytes, out={total_out_bytes} bytes ({:.2}× smaller)", total_in_bytes as f64 / total_out_bytes as f64);
    Ok(total_out_bytes)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_streaming_q4k_large_model")?;
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("model.q4k");
    let bytes_written = streaming_quantize_to_disk(&out)?;
    println!("wrote {bytes_written} bytes to {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn output_is_smaller_than_input() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("model.q4k");
        let bytes_out = streaming_quantize_to_disk(&out).unwrap();
        let bytes_in = TOTAL_PARAMS * 4; // f32 input
        assert!(
            bytes_out < bytes_in,
            "Q4K output {bytes_out} should be smaller than f32 input {bytes_in}"
        );
    }

    #[test]
    fn quantize_chunk_preserves_scale_header() {
        let chunk = vec![1.0, -2.0, 3.0, -4.0];
        let out = quantize_chunk_q4k(&chunk);
        let scale = f32::from_le_bytes(out[..4].try_into().unwrap());
        assert!((scale - 4.0).abs() < 0.01, "scale should be max-abs (4.0)");
    }
}
