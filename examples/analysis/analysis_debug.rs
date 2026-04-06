//! # Low-Level Model File Debug Inspection
//!
//! CLI equivalent: `apr debug model.apr`
//!
//! Parses raw APR model bytes to extract header fields: magic bytes, version,
//! flags (compressed, signed, encrypted), dtype, tensor count. Detects format
//! from magic bytes and produces an annotated hex dump.
//!
//! ## What this demonstrates
//! - Binary header parsing with explicit error handling
//! - Flag bitmask extraction (compressed, signed, encrypted)
//! - Format detection from magic bytes (APR2, GGUF, SafeTensors)
//! - Annotated hex dump of the first 64 bytes
//! - Graceful handling of corrupted files
//!
//!
//! ## Format Variants
//! ```bash
//! apr debug model.apr          # APR native format
//! apr debug model.gguf         # GGUF (llama.cpp compatible)
//! apr debug model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Detected container format based on magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatType {
    Apr,
    Gguf,
    SafeTensors,
    Unknown,
}

impl FormatType {
    fn label(self) -> &'static str {
        match self {
            Self::Apr => "APR v2",
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::Unknown => "Unknown",
        }
    }
}

/// Parsed header from the first bytes of a model file.
#[derive(Debug, Clone)]
struct HeaderInfo {
    magic: [u8; 4],
    version: u8,
    flags: u8,
    dtype: u8,
    tensor_count: u32,
    compressed: bool,
    signed: bool,
    encrypted: bool,
}

// Flag bit positions within the flags byte.
const FLAG_COMPRESSED: u8 = 0b0000_0001;
const FLAG_SIGNED: u8 = 0b0000_0010;
const FLAG_ENCRYPTED: u8 = 0b0000_0100;

// Minimum header size: 4 (magic) + 1 (version) + 1 (flags) + 1 (dtype) + 4 (tensor_count) = 11
const MIN_HEADER_SIZE: usize = 11;

// ---------------------------------------------------------------------------
// Parsing logic
// ---------------------------------------------------------------------------

/// Detect the container format from the first four magic bytes.
fn detect_format(magic: [u8; 4]) -> FormatType {
    match &magic {
        b"APR2" => FormatType::Apr,
        b"GGUF" => FormatType::Gguf,
        // SafeTensors files start with a little-endian u64 length prefix.
        // The first four bytes of a small header are typically a small number
        // followed by zeros, but we use a heuristic: if byte 0 is an opening
        // brace '{' it is likely JSON (SafeTensors header).
        _ if magic[0] == b'{' => FormatType::SafeTensors,
        _ => FormatType::Unknown,
    }
}

/// Parse header fields from raw bytes.
///
/// Layout (11 bytes minimum):
///   [0..4]  magic bytes
///   [4]     version
///   [5]     flags (bit 0 = compressed, bit 1 = signed, bit 2 = encrypted)
///   [6]     dtype (0=FP32, 1=FP16, 2=BF16, 3=INT8, 4=INT4)
///   [7..11] tensor_count (u32 LE)
fn parse_header(data: &[u8]) -> std::result::Result<HeaderInfo, String> {
    if data.len() < MIN_HEADER_SIZE {
        return Err(format!(
            "data too short for header: need {} bytes, got {}",
            MIN_HEADER_SIZE,
            data.len()
        ));
    }

    let mut magic = [0u8; 4];
    magic.copy_from_slice(&data[0..4]);

    let version = data[4];
    let flags = data[5];
    let dtype = data[6];

    let tensor_count = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);

    Ok(HeaderInfo {
        magic,
        version,
        flags,
        dtype,
        tensor_count,
        compressed: flags & FLAG_COMPRESSED != 0,
        signed: flags & FLAG_SIGNED != 0,
        encrypted: flags & FLAG_ENCRYPTED != 0,
    })
}

/// Human-readable label for a dtype byte.
fn dtype_label(dtype: u8) -> &'static str {
    match dtype {
        0 => "FP32",
        1 => "FP16",
        2 => "BF16",
        3 => "INT8",
        4 => "INT4",
        _ => "Unknown",
    }
}

// ---------------------------------------------------------------------------
// Hex dump
// ---------------------------------------------------------------------------

/// Produce a hex dump of up to `max_bytes`, annotating header fields.
fn hex_dump_annotated(data: &[u8], max_bytes: usize) -> String {
    let limit = data.len().min(max_bytes);
    let mut output = String::new();

    for row_start in (0..limit).step_by(16) {
        let row_end = (row_start + 16).min(limit);
        let hex_part: String = data[row_start..row_end]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");

        let ascii_part: String = data[row_start..row_end]
            .iter()
            .map(|&b| {
                if (0x20..=0x7e).contains(&b) {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();

        let annotation = row_annotation(row_start);

        output.push_str(&format!(
            "{row_start:08x}  {hex_part:<48}  |{ascii_part}|{annotation}\n",
        ));
    }
    output
}

/// Return an inline annotation for a given row offset.
fn row_annotation(row_start: usize) -> String {
    match row_start {
        0 => "  <- magic[0..4] version flags dtype tensor_count[7..11]".to_string(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

/// Print a formatted summary table of parsed header fields.
fn header_summary_table(header: &HeaderInfo, format: FormatType) -> String {
    let mut lines = Vec::with_capacity(10);
    lines.push("Field            Value".to_string());
    lines.push("-".repeat(40));

    let magic_str = String::from_utf8_lossy(&header.magic);
    lines.push(format!("Magic            {magic_str}"));
    lines.push(format!("Format           {}", format.label()));
    lines.push(format!("Version          {}", header.version));
    lines.push(format!("Flags byte       0b{:08b}", header.flags));
    lines.push(format!("  compressed     {}", header.compressed));
    lines.push(format!("  signed         {}", header.signed));
    lines.push(format!("  encrypted      {}", header.encrypted));
    lines.push(format!(
        "DType            {} ({})",
        header.dtype,
        dtype_label(header.dtype)
    ));
    lines.push(format!("Tensor count     {}", header.tensor_count));

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Synthetic model builder
// ---------------------------------------------------------------------------

/// Build a synthetic APR model file with the given parameters.
fn build_synthetic_model(rng: &mut impl Rng, tensor_count: u32, flags: u8, dtype: u8) -> Vec<u8> {
    let payload_size: usize = 64; // small payload per tensor
    let total_payload = payload_size * tensor_count as usize;
    let total_size = MIN_HEADER_SIZE + total_payload;

    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(b"APR2");
    data[4] = 2; // version
    data[5] = flags;
    data[6] = dtype;
    data[7..11].copy_from_slice(&tensor_count.to_le_bytes());

    // Random payload
    for byte in &mut data[MIN_HEADER_SIZE..] {
        *byte = rng.gen();
    }

    data
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_debug")?;

    // -- Section 1: Create synthetic APR model with known header ---------------
    println!("=== APR Debug: Low-Level File Inspection ===\n");

    let flags = FLAG_COMPRESSED | FLAG_SIGNED; // compressed + signed, not encrypted
    let dtype: u8 = 0; // FP32
    let tensor_count: u32 = 3;

    let model_data = build_synthetic_model(ctx.rng(), tensor_count, flags, dtype);
    println!(
        "Synthetic model: {} bytes, {} tensors\n",
        model_data.len(),
        tensor_count
    );

    // -- Section 2: Parse header -----------------------------------------------
    println!("--- Header Parse ---");
    let header = parse_header(&model_data).map_err(CookbookError::invalid_format)?;
    println!(
        "  magic:        {:?} ({})",
        header.magic,
        String::from_utf8_lossy(&header.magic)
    );
    println!("  version:      {}", header.version);
    println!("  flags:        0b{:08b}", header.flags);
    println!(
        "  dtype:        {} ({})",
        header.dtype,
        dtype_label(header.dtype)
    );
    println!("  tensor_count: {}", header.tensor_count);
    println!("  compressed:   {}", header.compressed);
    println!("  signed:       {}", header.signed);
    println!("  encrypted:    {}", header.encrypted);

    // -- Section 3: Format detection -------------------------------------------
    println!("\n--- Format Detection ---");

    let test_magics: &[([u8; 4], &str)] = &[
        (*b"APR2", "APR v2 model file"),
        (*b"GGUF", "GGUF quantized model"),
        (*b"{\"__", "SafeTensors JSON header"),
        (*b"\x00\x00\x00\x00", "Unknown / raw binary"),
    ];

    for (magic, description) in test_magics {
        let fmt = detect_format(*magic);
        println!(
            "  {:?} -> {:<14} ({})",
            String::from_utf8_lossy(magic),
            fmt.label(),
            description
        );
    }

    let detected = detect_format(header.magic);
    println!("\nDetected format for model: {}", detected.label());

    // -- Section 4: Hex dump first 64 bytes with annotations -------------------
    println!("\n--- Hex Dump (first 64 bytes) ---");
    let dump = hex_dump_annotated(&model_data, 64);
    print!("{dump}");

    // -- Section 5: Header field summary table ---------------------------------
    println!("\n--- Header Summary ---");
    let table = header_summary_table(&header, detected);
    println!("{table}");

    // -- Section 6: Debug corrupted file (bad magic) ---------------------------
    println!("\n--- Corrupted File Debug ---");
    let mut corrupted = model_data.clone();
    corrupted[0] = 0xFF;
    corrupted[1] = 0xFE;

    let corrupt_header = parse_header(&corrupted).map_err(CookbookError::invalid_format)?;
    let corrupt_fmt = detect_format(corrupt_header.magic);
    println!(
        "Corrupted magic: {:02x} {:02x} {:02x} {:02x}",
        corrupted[0], corrupted[1], corrupted[2], corrupted[3]
    );
    println!("Detected as:     {}", corrupt_fmt.label());
    println!("Valid APR:       {}", corrupt_fmt == FormatType::Apr);

    // Show hex dump of corrupted header
    println!("\nCorrupted hex (first 16 bytes):");
    let corrupt_dump = hex_dump_annotated(&corrupted, 16);
    print!("{corrupt_dump}");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_header(flags: u8, dtype: u8, tensor_count: u32) -> Vec<u8> {
        let mut data = vec![0u8; MIN_HEADER_SIZE + 64];
        data[0..4].copy_from_slice(b"APR2");
        data[4] = 2;
        data[5] = flags;
        data[6] = dtype;
        data[7..11].copy_from_slice(&tensor_count.to_le_bytes());
        data
    }

    #[test]
    fn test_parse_header_valid() {
        let data = make_valid_header(FLAG_COMPRESSED, 0, 5);
        let header = parse_header(&data);
        assert!(header.is_ok());
        let h = header.expect("valid header");
        assert_eq!(&h.magic, b"APR2");
        assert_eq!(h.version, 2);
        assert_eq!(h.tensor_count, 5);
        assert!(h.compressed);
        assert!(!h.signed);
        assert!(!h.encrypted);
    }

    #[test]
    fn test_parse_header_too_short() {
        let data = vec![0u8; 5];
        let result = parse_header(&data);
        assert!(result.is_err());
        let err = result.expect_err("should fail on short data");
        assert!(err.contains("too short"));
    }

    #[test]
    fn test_parse_header_all_flags() {
        let flags = FLAG_COMPRESSED | FLAG_SIGNED | FLAG_ENCRYPTED;
        let data = make_valid_header(flags, 1, 10);
        let h = parse_header(&data).expect("valid header");
        assert!(h.compressed);
        assert!(h.signed);
        assert!(h.encrypted);
        assert_eq!(h.dtype, 1);
    }

    #[test]
    fn test_parse_header_no_flags() {
        let data = make_valid_header(0, 0, 1);
        let h = parse_header(&data).expect("valid header");
        assert!(!h.compressed);
        assert!(!h.signed);
        assert!(!h.encrypted);
    }

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format(*b"APR2"), FormatType::Apr);
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format(*b"GGUF"), FormatType::Gguf);
    }

    #[test]
    fn test_detect_format_safetensors() {
        let magic = [b'{', b'"', b'_', b'_'];
        assert_eq!(detect_format(magic), FormatType::SafeTensors);
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format(*b"\x00\x00\x00\x00"), FormatType::Unknown);
        assert_eq!(detect_format(*b"XXXX"), FormatType::Unknown);
    }

    #[test]
    fn test_dtype_labels() {
        assert_eq!(dtype_label(0), "FP32");
        assert_eq!(dtype_label(1), "FP16");
        assert_eq!(dtype_label(2), "BF16");
        assert_eq!(dtype_label(3), "INT8");
        assert_eq!(dtype_label(4), "INT4");
        assert_eq!(dtype_label(255), "Unknown");
    }

    #[test]
    fn test_hex_dump_annotated_output() {
        let data = make_valid_header(FLAG_COMPRESSED, 0, 2);
        let dump = hex_dump_annotated(&data, 64);
        // Must contain the offset column
        assert!(dump.contains("00000000"));
        // Must contain the annotation for the first row
        assert!(dump.contains("magic"));
        // Must have ASCII column delimiters
        assert!(dump.contains('|'));
    }

    #[test]
    fn test_header_summary_table_content() {
        let data = make_valid_header(FLAG_COMPRESSED | FLAG_ENCRYPTED, 2, 7);
        let h = parse_header(&data).expect("valid header");
        let table = header_summary_table(&h, FormatType::Apr);
        assert!(table.contains("APR2"));
        assert!(table.contains("APR v2"));
        assert!(table.contains("compressed"));
        assert!(table.contains("true"));
        assert!(table.contains("BF16"));
        assert!(table.contains('7'));
    }
}
