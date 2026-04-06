#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: usize,
}

impl TensorInfo {
    pub fn param_count(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct InspectResult {
    pub name: String,
    pub description: String,
    pub format_version: u8,
    pub num_tensors: usize,
    pub total_params: usize,
    pub total_bytes: usize,
    pub compression: String,
    pub tensors: Vec<TensorInfo>,
}

impl fmt::Display for InspectResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model: {}", self.name)?;
        writeln!(f, "Description: {}", self.description)?;
        writeln!(f, "Format: APR v{}", self.format_version)?;
        writeln!(f, "Tensors: {}", self.num_tensors)?;
        writeln!(f, "Parameters: {}", self.total_params)?;
        writeln!(f, "Total size: {} bytes", self.total_bytes)?;
        writeln!(f, "Compression: {}", self.compression)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Inspection logic
// ---------------------------------------------------------------------------

pub fn inspect_apr(bytes: &[u8]) -> std::result::Result<InspectResult, String> {
    // Validate magic bytes
    if bytes.len() < 4 {
        return Err("File too small to be a valid APR model".to_string());
    }
    if &bytes[0..4] != b"APR2" {
        return Err(format!(
            "Invalid magic bytes: expected APR2, got {:?}",
            &bytes[0..4]
        ));
    }

    // Parse format version (byte 4)
    let format_version = if bytes.len() > 4 { bytes[4] } else { 2 };

    // Extract model name from metadata region
    let name = extract_metadata_string(bytes, "name").unwrap_or_else(|| "unknown".to_string());
    let description = extract_metadata_string(bytes, "description").unwrap_or_default();
    let compression = detect_compression(bytes);

    // Parse tensor directory
    let tensors = parse_tensor_directory(bytes);
    let num_tensors = tensors.len();
    let total_params: usize = tensors.iter().map(TensorInfo::param_count).sum();
    let total_bytes: usize = tensors.iter().map(|t| t.size_bytes).sum();

    Ok(InspectResult {
        name,
        description,
        format_version,
        num_tensors,
        total_params,
        total_bytes,
        compression,
        tensors,
    })
}

pub fn extract_metadata_string(bytes: &[u8], key: &str) -> Option<String> {
    // Search for key pattern in metadata region (after header)
    let search = format!("{}=", key);
    let search_bytes = search.as_bytes();
    for i in 4..bytes.len().saturating_sub(search_bytes.len()) {
        if &bytes[i..i + search_bytes.len()] == search_bytes {
            let start = i + search_bytes.len();
            let end = bytes[start..]
                .iter()
                .position(|&b| b == 0 || b == b'\n')
                .map_or(bytes.len().min(start + 256), |p| start + p);
            return Some(String::from_utf8_lossy(&bytes[start..end]).to_string());
        }
    }
    None
}

pub fn detect_compression(bytes: &[u8]) -> String {
    // Check for LZ4 frame magic (0x04224D18) in payload region
    let lz4_magic: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
    for window in bytes.windows(4) {
        if window == lz4_magic {
            return "LZ4".to_string();
        }
    }
    // Check for zstd magic (0xFD2FB528)
    let zstd_magic: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
    for window in bytes.windows(4) {
        if window == zstd_magic {
            return "Zstd".to_string();
        }
    }
    "None".to_string()
}

pub fn parse_tensor_directory(bytes: &[u8]) -> Vec<TensorInfo> {
    // For APR v2 bundles built with ModelBundleV2, extract tensor info
    // from the structured format. We simulate parsing here since the
    // actual format is handled by aprender.
    let mut tensors = Vec::new();
    let mut offset = 0;

    // Search for tensor markers in the payload
    let marker = b"tensor:";
    while offset < bytes.len().saturating_sub(marker.len() + 20) {
        if &bytes[offset..offset + marker.len()] == marker {
            let name_start = offset + marker.len();
            let name_end = bytes[name_start..]
                .iter()
                .position(|&b| b == b':')
                .map_or(name_start, |p| name_start + p);
            let name = String::from_utf8_lossy(&bytes[name_start..name_end]).to_string();

            // Parse shape from next segment
            let shape_start = name_end + 1;
            let shape_end = bytes[shape_start..]
                .iter()
                .position(|&b| b == b':')
                .map_or(shape_start, |p| shape_start + p);
            let shape_str = String::from_utf8_lossy(&bytes[shape_start..shape_end]);
            let shape: Vec<usize> = shape_str
                .split('x')
                .filter_map(|s| s.parse().ok())
                .collect();

            let param_count: usize = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };
            let size_bytes = param_count * 4; // assume FP32

            tensors.push(TensorInfo {
                name,
                shape,
                dtype: "f32".to_string(),
                size_bytes,
            });

            offset = shape_end + 1;
        } else {
            offset += 1;
        }
    }

    // If no tensors found via markers, infer from bundle size
    if tensors.is_empty() && bytes.len() > 64 {
        let payload_size = bytes.len().saturating_sub(64);
        let estimated_params = payload_size / 4;
        tensors.push(TensorInfo {
            name: "weights".to_string(),
            shape: vec![estimated_params],
            dtype: "f32".to_string(),
            size_bytes: payload_size,
        });
    }

    tensors
}

pub fn format_size(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

pub fn size_breakdown(tensors: &[TensorInfo]) -> HashMap<String, usize> {
    let mut breakdown = HashMap::new();
    for t in tensors {
        let category = if t.name.contains("embed") {
            "embedding"
        } else if t.name.contains("attn") || t.name.contains("attention") {
            "attention"
        } else if t.name.contains("ffn") || t.name.contains("mlp") {
            "feed-forward"
        } else if t.name.contains("norm") || t.name.contains("ln") {
            "normalization"
        } else {
            "other"
        };
        *breakdown.entry(category.to_string()).or_insert(0) += t.size_bytes;
    }
    breakdown
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
