//! # Multi-Step Conversion Chain
//!
//! **CLI equivalent:** `apr rosetta chain model.safetensors --through apr,gguf`
//!
//! Demonstrates chaining multiple format conversions in sequence.
//! Given a source file and a list of target formats, the chain converter
//! produces all intermediate representations along the way.
//!
//! ## Sections
//! 1. Chain planning — validate the sequence of formats
//! 2. Step-by-step conversion — execute each link in the chain
//! 3. Intermediate sizes — track size changes at each stage
//! 4. Final verification — confirm all outputs are valid
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Format types
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

/// Parse a format name string into a Format enum.
fn parse_format(s: &str) -> Option<Format> {
    match s.to_lowercase().as_str() {
        "apr" => Some(Format::Apr),
        "safetensors" | "st" => Some(Format::SafeTensors),
        "gguf" => Some(Format::Gguf),
        "onnx" => Some(Format::Onnx),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Chain types
// ---------------------------------------------------------------------------

/// A single link in the conversion chain with its output data.
#[derive(Debug)]
struct ChainLink {
    format: Format,
    data: Vec<u8>,
    size_bytes: usize,
}

/// Result of a chain conversion — all intermediate and final outputs.
struct ChainResult {
    links: Vec<ChainLink>,
    total_steps: usize,
}

// ---------------------------------------------------------------------------
// Format conversion simulation
// ---------------------------------------------------------------------------

/// Magic bytes / header for each format.
fn format_header(fmt: Format) -> Vec<u8> {
    match fmt {
        Format::Apr => b"APR2".to_vec(),
        Format::SafeTensors => {
            let header = b"{}";
            let mut v = Vec::new();
            v.extend_from_slice(&(header.len() as u64).to_le_bytes());
            v.extend_from_slice(header);
            v
        }
        Format::Gguf => {
            let mut v = Vec::new();
            v.extend_from_slice(b"GGUF");
            v.extend_from_slice(&3u32.to_le_bytes());
            v.extend_from_slice(&0u64.to_le_bytes()); // tensor count
            v.extend_from_slice(&0u64.to_le_bytes()); // metadata count
            v
        }
        Format::Onnx => {
            let mut v = Vec::new();
            v.extend_from_slice(b"\x08\x07"); // ONNX IR version marker
            v
        }
    }
}

/// Strip the format-specific header from data, returning raw payload.
fn strip_header(data: &[u8], fmt: Format) -> Vec<u8> {
    let skip = match fmt {
        Format::Apr => 4,
        Format::SafeTensors => {
            if data.len() >= 8 {
                let header_len =
                    u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
                8 + header_len
            } else {
                0
            }
        }
        Format::Gguf => 24, // magic(4) + version(4) + tc(8) + kvc(8)
        Format::Onnx => 2,
    };
    data.get(skip..).unwrap_or(&[]).to_vec()
}

/// Convert data from one format to another.
fn convert_single(data: &[u8], from: Format, to: Format) -> Vec<u8> {
    if from == to {
        return data.to_vec();
    }
    let raw = strip_header(data, from);
    let mut output = format_header(to);
    output.extend_from_slice(&raw);
    output
}

// ---------------------------------------------------------------------------
// Chain conversion
// ---------------------------------------------------------------------------

/// Execute a chain conversion through a sequence of formats.
///
/// Given source data in `start_format`, converts through each format in
/// `chain` sequentially. Returns all intermediate outputs.
///
/// An empty chain returns the source data unchanged.
fn chain_convert(data: &[u8], start_format: Format, chain: &[Format]) -> ChainResult {
    let mut links = Vec::new();

    // Always record the source
    links.push(ChainLink {
        format: start_format,
        data: data.to_vec(),
        size_bytes: data.len(),
    });

    let mut current_data = data.to_vec();
    let mut current_format = start_format;

    for &next_format in chain {
        let converted = convert_single(&current_data, current_format, next_format);
        links.push(ChainLink {
            format: next_format,
            data: converted.clone(),
            size_bytes: converted.len(),
        });
        current_data = converted;
        current_format = next_format;
    }

    ChainResult {
        total_steps: chain.len(),
        links,
    }
}

/// Validate that chain output has correct format markers.
fn validate_chain_output(link: &ChainLink) -> bool {
    match link.format {
        Format::Apr => link.data.len() >= 4 && &link.data[0..4] == b"APR2",
        Format::Gguf => link.data.len() >= 4 && &link.data[0..4] == b"GGUF",
        Format::SafeTensors => {
            if link.data.len() < 8 {
                return false;
            }
            let header_len = u64::from_le_bytes(link.data[0..8].try_into().unwrap_or([0; 8]));
            link.data.len() >= 8 + header_len as usize
        }
        Format::Onnx => link.data.len() >= 2 && link.data[0] == 0x08,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_rosetta_chain")?;

    // Section 1: Chain planning
    println!("=== Chain Planning ===");
    let chain_spec = "apr,gguf,safetensors";
    let chain_formats: Vec<Format> = chain_spec
        .split(',')
        .filter_map(|s| parse_format(s.trim()))
        .collect();
    println!(
        "Chain: SafeTensors → {}",
        chain_formats
            .iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(" → ")
    );
    println!("Total steps: {}", chain_formats.len());
    println!();

    // Section 2: Step-by-step conversion
    println!("=== Step-by-Step Conversion ===");

    // Create source SafeTensors data
    let source_payload = generate_model_payload(42, 4096);
    let mut source_data = Vec::new();
    let header = b"{\"w\":{\"dtype\":\"F32\",\"shape\":[32,32],\"data_offsets\":[0,4096]}}";
    source_data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source_data.extend_from_slice(header);
    source_data.extend_from_slice(&source_payload);

    let result = chain_convert(&source_data, Format::SafeTensors, &chain_formats);

    for (i, link) in result.links.iter().enumerate() {
        let label = if i == 0 { "Source" } else { "Step" };
        let valid = validate_chain_output(link);
        println!(
            "  {label} {i}: {:<15} {:>8} bytes  valid={}",
            format!("{}", link.format),
            link.size_bytes,
            valid,
        );
    }
    println!();

    // Section 3: Intermediate sizes
    println!("=== Size Comparison ===");
    println!(
        "{:<5} {:<15} {:<12} {:<10}",
        "Step", "Format", "Size (B)", "Delta"
    );
    println!("{}", "-".repeat(42));
    for (i, link) in result.links.iter().enumerate() {
        let delta = if i == 0 {
            "baseline".to_string()
        } else {
            let prev = result.links[i - 1].size_bytes as i64;
            let curr = link.size_bytes as i64;
            let diff = curr - prev;
            format!("{:+}", diff)
        };
        println!(
            "{:<5} {:<15} {:<12} {:<10}",
            i,
            format!("{}", link.format),
            link.size_bytes,
            delta,
        );
    }
    println!();

    // Section 4: Final verification
    println!("=== Final Verification ===");
    let final_link = result.links.last().unwrap();
    let valid = validate_chain_output(final_link);
    println!("Final format:  {}", final_link.format);
    println!("Final size:    {} bytes", final_link.size_bytes);
    println!("Valid:         {valid}");
    println!("Total steps:   {}", result.total_steps);
    assert!(valid, "Final output must be valid");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_apr_source(size: usize) -> Vec<u8> {
        let mut data = b"APR2".to_vec();
        data.extend_from_slice(&generate_model_payload(42, size));
        data
    }

    #[test]
    fn test_empty_chain_is_identity() {
        let source = make_apr_source(256);
        let result = chain_convert(&source, Format::Apr, &[]);
        assert_eq!(result.total_steps, 0);
        assert_eq!(result.links.len(), 1); // just the source
        assert_eq!(result.links[0].data, source);
    }

    #[test]
    fn test_single_step_chain() {
        let source = make_apr_source(256);
        let result = chain_convert(&source, Format::Apr, &[Format::Gguf]);
        assert_eq!(result.total_steps, 1);
        assert_eq!(result.links.len(), 2);
        assert_eq!(result.links[1].format, Format::Gguf);
    }

    #[test]
    fn test_multi_step_chain() {
        let source = make_apr_source(256);
        let chain = vec![Format::Gguf, Format::Apr, Format::SafeTensors];
        let result = chain_convert(&source, Format::Apr, &chain);
        assert_eq!(result.total_steps, 3);
        assert_eq!(result.links.len(), 4);
        assert_eq!(result.links[1].format, Format::Gguf);
        assert_eq!(result.links[2].format, Format::Apr);
        assert_eq!(result.links[3].format, Format::SafeTensors);
    }

    #[test]
    fn test_chain_produces_valid_output() {
        let source = make_apr_source(512);
        let chain = vec![Format::Gguf, Format::Apr];
        let result = chain_convert(&source, Format::Apr, &chain);
        for link in &result.links {
            assert!(
                validate_chain_output(link),
                "Invalid output for {}",
                link.format
            );
        }
    }

    #[test]
    fn test_same_format_chain_preserves_data() {
        let source = make_apr_source(128);
        let result = chain_convert(&source, Format::Apr, &[Format::Apr]);
        assert_eq!(result.links[0].data, result.links[1].data);
    }

    #[test]
    fn test_chain_sizes_all_positive() {
        let source = make_apr_source(1024);
        let chain = vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Apr];
        let result = chain_convert(&source, Format::Apr, &chain);
        for link in &result.links {
            assert!(link.size_bytes > 0, "Size must be positive");
        }
    }

    #[test]
    fn test_parse_format_variants() {
        assert_eq!(parse_format("apr"), Some(Format::Apr));
        assert_eq!(parse_format("APR"), Some(Format::Apr));
        assert_eq!(parse_format("safetensors"), Some(Format::SafeTensors));
        assert_eq!(parse_format("ST"), Some(Format::SafeTensors));
        assert_eq!(parse_format("gguf"), Some(Format::Gguf));
        assert_eq!(parse_format("onnx"), Some(Format::Onnx));
        assert_eq!(parse_format("unknown"), None);
    }

    #[test]
    fn test_validate_apr_output() {
        let link = ChainLink {
            format: Format::Apr,
            data: b"APR2hello".to_vec(),
            size_bytes: 9,
        };
        assert!(validate_chain_output(&link));
    }

    #[test]
    fn test_validate_invalid_apr() {
        let link = ChainLink {
            format: Format::Apr,
            data: b"NOPE".to_vec(),
            size_bytes: 4,
        };
        assert!(!validate_chain_output(&link));
    }

    #[test]
    fn test_strip_and_reapply_header() {
        let source = make_apr_source(256);
        let raw = strip_header(&source, Format::Apr);
        let rebuilt = convert_single(&source, Format::Apr, Format::Apr);
        let rebuilt_raw = strip_header(&rebuilt, Format::Apr);
        assert_eq!(raw, rebuilt_raw);
    }
}
