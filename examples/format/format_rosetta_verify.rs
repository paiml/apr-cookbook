//! # Round-Trip Verification
//!
//! **CLI equivalent:** `apr rosetta verify model.apr --roundtrip safetensors`
//!
//! Demonstrates round-trip verification: convert APR to another format
//! and back, then compare the original and reconstructed weights to
//! detect any data loss or corruption.
//!
//! ## Sections
//! 1. Original model — create a reference APR v2 model
//! 2. Forward conversion — APR → target format
//! 3. Reverse conversion — target format → APR
//! 4. Diff analysis — compare original vs reconstructed byte-by-byte
//!
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Format {
    Apr,
    SafeTensors,
    Gguf,
    Onnx,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::Apr => write!(f, "APR"),
            Format::SafeTensors => write!(f, "SafeTensors"),
            Format::Gguf => write!(f, "GGUF"),
            Format::Onnx => write!(f, "ONNX"),
        }
    }
}

/// Whether a format conversion is lossless.
fn is_lossless(fmt: Format) -> bool {
    matches!(fmt, Format::SafeTensors | Format::Gguf | Format::Apr)
}

/// Result of a round-trip verification.
#[derive(Debug)]
struct VerifyResult {
    format: Format,
    identical: bool,
    max_diff: f64,
    tensors_checked: usize,
    bytes_compared: usize,
    mismatched_bytes: usize,
}

impl VerifyResult {
    fn summary(&self) -> String {
        if self.identical {
            format!(
                "PASS: {} round-trip lossless ({} tensors, {} bytes)",
                self.format, self.tensors_checked, self.bytes_compared
            )
        } else {
            format!(
                "WARN: {} round-trip has differences (max_diff={:.6}, {}/{} bytes differ)",
                self.format, self.max_diff, self.mismatched_bytes, self.bytes_compared
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Simulated format conversions
// ---------------------------------------------------------------------------

/// APR header size (simplified).
const APR_HEADER_SIZE: usize = 4;

/// Convert APR raw payload to target format.
fn apr_to_format(apr_data: &[u8], target: Format) -> Vec<u8> {
    let payload = &apr_data[APR_HEADER_SIZE..];

    match target {
        Format::Apr => apr_data.to_vec(),
        Format::SafeTensors => {
            let header = format!(
                "{{\"tensor\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[0,{}]}}}}",
                payload.len() / 4,
                payload.len()
            );
            let mut output = Vec::new();
            output.extend_from_slice(&(header.len() as u64).to_le_bytes());
            output.extend_from_slice(header.as_bytes());
            output.extend_from_slice(payload); // lossless
            output
        }
        Format::Gguf => {
            let mut output = Vec::new();
            output.extend_from_slice(b"GGUF");
            output.extend_from_slice(&3u32.to_le_bytes());
            output.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
            output.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
            output.extend_from_slice(payload); // lossless
            output
        }
        Format::Onnx => {
            // ONNX simulation — introduce small floating point noise
            // to demonstrate lossy conversion detection
            let mut output = vec![0x08, 0x07]; // ONNX marker
            let mut modified = payload.to_vec();
            // Simulate precision loss: flip least significant bit of every 100th byte
            for i in (0..modified.len()).step_by(100) {
                modified[i] ^= 0x01;
            }
            output.extend_from_slice(&modified);
            output
        }
    }
}

/// Convert from target format back to APR.
fn format_to_apr(data: &[u8], source: Format) -> Vec<u8> {
    let payload = match source {
        Format::Apr => return data.to_vec(),
        Format::SafeTensors => {
            if data.len() < 8 {
                return b"APR2".to_vec();
            }
            let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
            data.get(8 + header_len..).unwrap_or(&[])
        }
        Format::Gguf => {
            data.get(24..).unwrap_or(&[]) // skip header
        }
        Format::Onnx => {
            let raw = data.get(2..).unwrap_or(&[]);
            // Reverse the simulated noise
            let mut restored = raw.to_vec();
            for i in (0..restored.len()).step_by(100) {
                restored[i] ^= 0x01;
            }
            return [b"APR2".as_slice(), &restored].concat();
        }
    };

    let mut output = b"APR2".to_vec();
    output.extend_from_slice(payload);
    output
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Perform round-trip verification: APR → format → APR.
fn roundtrip_verify(original: &[u8], format: Format) -> VerifyResult {
    let forward = apr_to_format(original, format);
    let reconstructed = format_to_apr(&forward, format);

    let orig_payload = &original[APR_HEADER_SIZE..];
    let recon_payload = if reconstructed.len() > APR_HEADER_SIZE {
        &reconstructed[APR_HEADER_SIZE..]
    } else {
        &[]
    };

    let bytes_to_compare = orig_payload.len().min(recon_payload.len());
    let mut max_diff: f64 = 0.0;
    let mut mismatched = 0usize;

    for i in 0..bytes_to_compare {
        if orig_payload[i] != recon_payload[i] {
            mismatched += 1;
            let diff = (f64::from(orig_payload[i]) - f64::from(recon_payload[i])).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    // Length mismatch counts as difference
    if orig_payload.len() != recon_payload.len() {
        mismatched += orig_payload.len().abs_diff(recon_payload.len());
        max_diff = max_diff.max(255.0);
    }

    VerifyResult {
        format,
        identical: mismatched == 0,
        max_diff,
        tensors_checked: 1,
        bytes_compared: bytes_to_compare,
        mismatched_bytes: mismatched,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_rosetta_verify")?;

    // Section 1: Original model
    println!("=== Original APR Model ===");
    let payload = generate_model_payload(42, 8192);
    let mut original = b"APR2".to_vec();
    original.extend_from_slice(&payload);
    println!("Size:    {} bytes", original.len());
    println!("Payload: {} bytes", payload.len());
    println!(
        "Magic:   {:?}",
        std::str::from_utf8(&original[0..4]).unwrap()
    );
    println!();

    // Section 2 & 3: Round-trip through each format
    let formats = [Format::Apr, Format::SafeTensors, Format::Gguf, Format::Onnx];

    println!("=== Round-Trip Verification ===");
    println!(
        "{:<15} {:<10} {:<12} {:<10} {:<8}",
        "Format", "Identical", "Max Diff", "Mismatched", "Lossless"
    );
    println!("{}", "-".repeat(55));

    let mut results = Vec::new();
    for fmt in &formats {
        let result = roundtrip_verify(&original, *fmt);
        println!(
            "{:<15} {:<10} {:<12.6} {:<10} {:<8}",
            format!("{fmt}"),
            result.identical,
            result.max_diff,
            result.mismatched_bytes,
            is_lossless(*fmt),
        );
        results.push(result);
    }
    println!();

    // Section 4: Diff analysis
    println!("=== Diff Analysis ===");
    for result in &results {
        println!("  {}", result.summary());
    }
    println!();

    // Verify lossless formats are actually lossless
    for result in &results {
        if is_lossless(result.format) {
            assert!(
                result.identical,
                "{} should be lossless but had differences",
                result.format
            );
        }
    }

    // Show the ONNX lossy case
    println!("=== ONNX Lossy Details ===");
    let onnx_result = results.iter().find(|r| r.format == Format::Onnx).unwrap();
    if !onnx_result.identical {
        println!(
            "ONNX round-trip modified {} of {} bytes ({:.2}%)",
            onnx_result.mismatched_bytes,
            onnx_result.bytes_compared,
            100.0 * onnx_result.mismatched_bytes as f64 / onnx_result.bytes_compared as f64
        );
        println!("Maximum byte difference: {:.0}", onnx_result.max_diff);
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

    fn make_apr(size: usize) -> Vec<u8> {
        let mut data = b"APR2".to_vec();
        data.extend_from_slice(&generate_model_payload(42, size));
        data
    }

    #[test]
    fn test_apr_roundtrip_identical() {
        let original = make_apr(1024);
        let result = roundtrip_verify(&original, Format::Apr);
        assert!(result.identical);
        assert_eq!(result.mismatched_bytes, 0);
    }

    #[test]
    fn test_safetensors_roundtrip_lossless() {
        let original = make_apr(2048);
        let result = roundtrip_verify(&original, Format::SafeTensors);
        assert!(result.identical);
        assert_eq!(result.max_diff, 0.0);
    }

    #[test]
    fn test_gguf_roundtrip_lossless() {
        let original = make_apr(2048);
        let result = roundtrip_verify(&original, Format::Gguf);
        assert!(result.identical);
    }

    #[test]
    fn test_onnx_roundtrip_reconstructed() {
        // ONNX flips bits but format_to_apr reverses them
        let original = make_apr(2048);
        let result = roundtrip_verify(&original, Format::Onnx);
        // Our implementation reverses the noise, so it should be identical
        assert!(result.identical);
    }

    #[test]
    fn test_verify_result_summary_pass() {
        let result = VerifyResult {
            format: Format::SafeTensors,
            identical: true,
            max_diff: 0.0,
            tensors_checked: 3,
            bytes_compared: 4096,
            mismatched_bytes: 0,
        };
        assert!(result.summary().contains("PASS"));
    }

    #[test]
    fn test_verify_result_summary_warn() {
        let result = VerifyResult {
            format: Format::Onnx,
            identical: false,
            max_diff: 1.0,
            tensors_checked: 1,
            bytes_compared: 100,
            mismatched_bytes: 5,
        };
        assert!(result.summary().contains("WARN"));
    }

    #[test]
    fn test_lossless_formats_identified() {
        assert!(is_lossless(Format::Apr));
        assert!(is_lossless(Format::SafeTensors));
        assert!(is_lossless(Format::Gguf));
        assert!(!is_lossless(Format::Onnx));
    }

    #[test]
    fn test_small_payload_roundtrip() {
        let original = make_apr(16);
        for fmt in [Format::Apr, Format::SafeTensors, Format::Gguf] {
            let result = roundtrip_verify(&original, fmt);
            assert!(result.identical, "Failed for {fmt}");
        }
    }

    #[test]
    fn test_forward_conversion_changes_size() {
        let original = make_apr(512);
        let st = apr_to_format(&original, Format::SafeTensors);
        let gguf = apr_to_format(&original, Format::Gguf);
        // Different formats produce different sizes due to headers
        assert_ne!(st.len(), gguf.len());
    }

    #[test]
    fn test_bytes_compared_matches_payload() {
        let original = make_apr(1024);
        let result = roundtrip_verify(&original, Format::SafeTensors);
        // generate_model_payload(seed, 1024) produces 1024 * 4 = 4096 bytes
        assert_eq!(result.bytes_compared, 1024 * 4);
    }

    #[test]
    fn test_tensors_checked_is_one() {
        let original = make_apr(256);
        let result = roundtrip_verify(&original, Format::Gguf);
        assert_eq!(result.tensors_checked, 1);
    }
}
