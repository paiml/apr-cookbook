//! # Format-Aware Binary Forensics
//!
//! Hex dump with APR format annotations, parsing magic bytes, version,
//! metadata offsets, and tensor data regions.
//!
//! ## CLI equivalent
//! ```bash
//! apr hex model.apr
//! ```
//!
//! ## What this demonstrates
//! - APR v2 binary format structure parsing
//! - Annotated hex dump with region labels
//! - Magic byte identification and format validation
//! - Offset calculation for format regions

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HexAnnotation {
    offset: usize,
    length: usize,
    label: String,
    value: String,
}

#[derive(Debug, Clone)]
struct FormatStructure {
    magic: [u8; 4],
    version: u32,
    metadata_offset: u32,
    tensor_data_offset: u32,
    total_size: usize,
}

// ---------------------------------------------------------------------------
// Hex dump and annotation
// ---------------------------------------------------------------------------

/// Read a little-endian u32 from a byte slice.
fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Header field descriptor for fixed-width u32 fields.
struct HeaderField {
    offset: usize,
    label: &'static str,
    format_value: fn(u32) -> String,
}

/// Format a u32 value as a version string (e.g., "v2").
fn fmt_version(v: u32) -> String {
    format!("v{v}")
}

/// Format a u32 value as a byte offset (e.g., "byte 64").
fn fmt_byte_offset(v: u32) -> String {
    format!("byte {v}")
}

/// List of fixed-width u32 header fields in the APR v2 format.
const HEADER_FIELDS: &[HeaderField] = &[
    HeaderField {
        offset: 4,
        label: "format version",
        format_value: fmt_version,
    },
    HeaderField {
        offset: 8,
        label: "metadata offset",
        format_value: fmt_byte_offset,
    },
    HeaderField {
        offset: 12,
        label: "tensor data offset",
        format_value: fmt_byte_offset,
    },
];

/// Annotate fixed-width u32 header fields at known offsets.
fn annotate_header_fields(data: &[u8], limit: usize) -> Vec<HexAnnotation> {
    HEADER_FIELDS
        .iter()
        .filter(|f| limit >= f.offset + 4)
        .filter_map(|f| {
            read_u32_le(data, f.offset).map(|v| HexAnnotation {
                offset: f.offset,
                length: 4,
                label: f.label.to_string(),
                value: format!(
                    "{} ({})",
                    bytes_to_hex(&data[f.offset..f.offset + 4]),
                    (f.format_value)(v)
                ),
            })
        })
        .collect()
}

/// Annotate variable-length regions: header/reserved, metadata, and tensor data.
fn annotate_variable_regions(data: &[u8], limit: usize) -> Vec<HexAnnotation> {
    let mut annotations = Vec::new();
    let meta_off = if limit >= 12 {
        read_u32_le(data, 8)
    } else {
        None
    };
    let tensor_off = if limit >= 16 {
        read_u32_le(data, 12)
    } else {
        None
    };

    // Header region (16..metadata_offset or 16..64)
    let header_end = meta_off.map_or(limit.min(64), |v| (v as usize).min(limit));
    if header_end > 16 && limit > 16 {
        let region_end = header_end.min(limit);
        annotations.push(HexAnnotation {
            offset: 16,
            length: region_end - 16,
            label: "header / reserved".to_string(),
            value: format!("{} bytes", region_end - 16),
        });
    }

    // Metadata region
    if let Some(mo) = meta_off {
        let meta_start = mo as usize;
        let meta_end = tensor_off.map_or(limit, |v| (v as usize).min(limit));
        if meta_start < limit && meta_start < meta_end {
            annotations.push(HexAnnotation {
                offset: meta_start,
                length: meta_end.min(limit) - meta_start,
                label: "metadata region".to_string(),
                value: format!("{} bytes", meta_end.min(limit) - meta_start),
            });
        }
    }

    // Tensor data region
    if let Some(to) = tensor_off {
        let tensor_start = to as usize;
        if tensor_start < limit {
            annotations.push(HexAnnotation {
                offset: tensor_start,
                length: limit - tensor_start,
                label: "tensor data region".to_string(),
                value: format!("{} bytes", limit - tensor_start),
            });
        }
    }

    annotations
}

/// Produce annotated hex dump of APR v2 format data.
fn annotated_hex_dump(data: &[u8], max_bytes: usize) -> Vec<HexAnnotation> {
    let limit = data.len().min(max_bytes);

    if limit < 4 {
        return if data.is_empty() {
            Vec::new()
        } else {
            vec![HexAnnotation {
                offset: 0,
                length: limit,
                label: "incomplete data".to_string(),
                value: bytes_to_hex(&data[..limit]),
            }]
        };
    }

    // Magic bytes (offset 0-3)
    let magic = &data[0..4];
    let magic_str = String::from_utf8_lossy(magic).to_string();
    let mut annotations = vec![HexAnnotation {
        offset: 0,
        length: 4,
        label: "magic bytes".to_string(),
        value: format!("{} ({})", bytes_to_hex(magic), magic_str),
    }];

    annotations.extend(annotate_header_fields(data, limit));
    annotations.extend(annotate_variable_regions(data, limit));
    annotations
}

/// Parse the APR v2 format structure from raw bytes.
fn parse_format_structure(data: &[u8]) -> Option<FormatStructure> {
    if data.len() < 16 {
        return None;
    }

    let mut magic = [0u8; 4];
    magic.copy_from_slice(&data[0..4]);

    Some(FormatStructure {
        magic,
        version: read_u32_le(data, 4)?,
        metadata_offset: read_u32_le(data, 8)?,
        tensor_data_offset: read_u32_le(data, 12)?,
        total_size: data.len(),
    })
}

/// Convert bytes to hex string.
fn bytes_to_hex(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render a classic hex dump view of data.
fn hex_dump_view(data: &[u8], max_bytes: usize) -> String {
    let mut output = String::new();
    let limit = data.len().min(max_bytes);

    for offset in (0..limit).step_by(16) {
        let end = (offset + 16).min(limit);
        let hex: Vec<String> = data[offset..end]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let ascii: String = data[offset..end]
            .iter()
            .map(|&b| {
                if (0x20..=0x7e).contains(&b) {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();

        output.push_str(&format!(
            "{:08x}  {:<48}  |{}|\n",
            offset,
            hex.join(" "),
            ascii,
        ));
    }

    output
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_hex")?;

    // ── Section 1: Build synthetic APR v2 binary ────────────────────────
    println!("=== APR Format-Aware Hex Dump ===\n");

    let metadata = b"test-model\x00fp32\x00lz4\x00";
    let tensor_data: Vec<u8> = (0..64).map(|i| (i * 7 + 13) as u8).collect();

    let metadata_offset: u32 = 64;
    let tensor_data_offset: u32 = metadata_offset + metadata.len() as u32;
    let total_size = tensor_data_offset as usize + tensor_data.len();

    let mut binary = vec![0u8; total_size];
    // Magic: APR2
    binary[0..4].copy_from_slice(b"APR2");
    // Version: 2
    binary[4..8].copy_from_slice(&2u32.to_le_bytes());
    // Metadata offset
    binary[8..12].copy_from_slice(&metadata_offset.to_le_bytes());
    // Tensor data offset
    binary[12..16].copy_from_slice(&tensor_data_offset.to_le_bytes());
    // Metadata
    binary[metadata_offset as usize..tensor_data_offset as usize].copy_from_slice(metadata);
    // Tensor data
    binary[tensor_data_offset as usize..].copy_from_slice(&tensor_data);

    println!("Binary size: {} bytes", binary.len());

    // ── Section 2: Raw hex view ─────────────────────────────────────────
    println!("\n--- Raw Hex View ---");
    let hex_view = hex_dump_view(&binary, 128);
    print!("{}", hex_view);

    // ── Section 3: Annotated regions ────────────────────────────────────
    println!("--- Annotated Regions ---");
    let annotations = annotated_hex_dump(&binary, binary.len());
    for ann in &annotations {
        println!(
            "  [{:04x}..{:04x}] {} = {}",
            ann.offset,
            ann.offset + ann.length,
            ann.label,
            ann.value,
        );
    }

    // ── Section 4: Magic byte identification ────────────────────────────
    println!("\n--- Magic Byte Identification ---");
    let magic = &binary[0..4];
    let is_apr2 = magic == b"APR2";
    println!(
        "Magic: {} -> {}",
        bytes_to_hex(magic),
        if is_apr2 {
            "APR v2 format"
        } else {
            "Unknown format"
        }
    );

    // ── Section 5: Format structure map ─────────────────────────────────
    println!("\n--- Format Structure Map ---");
    if let Some(structure) = parse_format_structure(&binary) {
        println!(
            "  Magic:              {:?}",
            String::from_utf8_lossy(&structure.magic)
        );
        println!("  Version:            {}", structure.version);
        println!(
            "  Metadata offset:    {} (0x{:04x})",
            structure.metadata_offset, structure.metadata_offset
        );
        println!(
            "  Tensor data offset: {} (0x{:04x})",
            structure.tensor_data_offset, structure.tensor_data_offset
        );
        println!("  Total size:         {} bytes", structure.total_size);
    }

    // Fingerprint
    let mut hasher = DefaultHasher::new();
    binary.len().hash(&mut hasher);
    annotations.len().hash(&mut hasher);
    println!("\nHex dump fingerprint: {:016x}", hasher.finish());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_apr2_binary(size: usize) -> Vec<u8> {
        let mut data = vec![0u8; size.max(16)];
        data[0..4].copy_from_slice(b"APR2");
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        data[8..12].copy_from_slice(&64u32.to_le_bytes()); // metadata at 64
        data[12..16].copy_from_slice(&80u32.to_le_bytes()); // tensor data at 80
        data
    }

    #[test]
    fn test_magic_bytes_annotated() {
        let data = make_apr2_binary(128);
        let annotations = annotated_hex_dump(&data, 128);
        let magic_ann = annotations.iter().find(|a| a.label == "magic bytes");
        assert!(magic_ann.is_some());
        let ann = magic_ann.unwrap();
        assert_eq!(ann.offset, 0);
        assert_eq!(ann.length, 4);
        assert!(ann.value.contains("APR2"));
    }

    #[test]
    fn test_version_annotated() {
        let data = make_apr2_binary(128);
        let annotations = annotated_hex_dump(&data, 128);
        let ver_ann = annotations.iter().find(|a| a.label == "format version");
        assert!(ver_ann.is_some());
        assert!(ver_ann.unwrap().value.contains("v2"));
    }

    #[test]
    fn test_offset_calculations() {
        let data = make_apr2_binary(128);
        let annotations = annotated_hex_dump(&data, 128);

        let meta_ann = annotations.iter().find(|a| a.label == "metadata offset");
        assert!(meta_ann.is_some());
        assert!(meta_ann.unwrap().value.contains("byte 64"));

        let tensor_ann = annotations.iter().find(|a| a.label == "tensor data offset");
        assert!(tensor_ann.is_some());
        assert!(tensor_ann.unwrap().value.contains("byte 80"));
    }

    #[test]
    fn test_handles_short_data() {
        let data = vec![0x41u8, 0x50, 0x52]; // only 3 bytes
        let annotations = annotated_hex_dump(&data, 10);
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].label, "incomplete data");
    }

    #[test]
    fn test_handles_empty_data() {
        let data: Vec<u8> = vec![];
        let annotations = annotated_hex_dump(&data, 10);
        assert!(annotations.is_empty());
    }

    #[test]
    fn test_annotations_non_overlapping() {
        let data = make_apr2_binary(128);
        let annotations = annotated_hex_dump(&data, 128);

        // Check no two annotations overlap (within the header region)
        let header_anns: Vec<&HexAnnotation> =
            annotations.iter().filter(|a| a.offset < 16).collect();

        for i in 0..header_anns.len() {
            for j in (i + 1)..header_anns.len() {
                let a = header_anns[i];
                let b = header_anns[j];
                let a_end = a.offset + a.length;
                let b_end = b.offset + b.length;
                assert!(
                    a_end <= b.offset || b_end <= a.offset,
                    "annotations overlap: [{}-{}] and [{}-{}]",
                    a.offset,
                    a_end,
                    b.offset,
                    b_end,
                );
            }
        }
    }

    #[test]
    fn test_parse_format_structure() {
        let data = make_apr2_binary(128);
        let structure = parse_format_structure(&data);
        assert!(structure.is_some());
        let s = structure.unwrap();
        assert_eq!(&s.magic, b"APR2");
        assert_eq!(s.version, 2);
        assert_eq!(s.metadata_offset, 64);
        assert_eq!(s.tensor_data_offset, 80);
    }

    #[test]
    fn test_parse_format_structure_too_short() {
        let data = vec![0u8; 10];
        assert!(parse_format_structure(&data).is_none());
    }

    #[test]
    fn test_bytes_to_hex() {
        assert_eq!(bytes_to_hex(&[0x41, 0x50, 0x52, 0x32]), "41 50 52 32");
        assert_eq!(bytes_to_hex(&[0x00, 0xff]), "00 ff");
    }

    #[test]
    fn test_hex_dump_view_format() {
        let data = make_apr2_binary(32);
        let view = hex_dump_view(&data, 32);
        assert!(view.contains("00000000"));
        assert!(view.contains("00000010"));
        // Should contain ASCII representation
        assert!(view.contains("|APR2"));
    }

    #[test]
    fn test_read_u32_le() {
        let data = [0x01, 0x00, 0x00, 0x00]; // 1 in LE
        assert_eq!(read_u32_le(&data, 0), Some(1));

        let data2 = [0x00, 0x01]; // too short
        assert_eq!(read_u32_le(&data2, 0), None);
    }
}
