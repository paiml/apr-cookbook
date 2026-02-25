//! # Cross-Format Conversion via Rosetta
//!
//! **CLI equivalent:** `apr rosetta convert --from safetensors --to apr`
//!
//! Demonstrates cross-format conversion using an intermediate representation.
//! The Rosetta module finds the optimal conversion path between any two
//! supported formats and executes the transformation step by step.
//!
//! ## Sections
//! 1. Format registry — supported formats and their capabilities
//! 2. Path finding — discover direct and transitive conversion paths
//! 3. Conversion execution — apply each step in the conversion path
//! 4. Verification — validate the output matches expectations

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Format types
// ---------------------------------------------------------------------------

/// Supported model formats.
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

/// A single conversion step.
#[derive(Debug, Clone)]
struct ConversionStep {
    from: Format,
    to: Format,
    lossy: bool,
    description: String,
}

/// A complete conversion path from source to target format.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ConversionPath {
    from: Format,
    to: Format,
    steps: Vec<ConversionStep>,
}

impl ConversionPath {
    fn is_direct(&self) -> bool {
        self.steps.len() == 1
    }

    fn is_identity(&self) -> bool {
        self.steps.is_empty()
    }

    fn is_lossy(&self) -> bool {
        self.steps.iter().any(|s| s.lossy)
    }
}

/// Format registry with conversion capabilities.
struct FormatRegistry {
    /// Adjacency list: from → [(to, lossy, description)]
    conversions: HashMap<Format, Vec<(Format, bool, String)>>,
}

// ---------------------------------------------------------------------------
// Registry and path finding
// ---------------------------------------------------------------------------

impl FormatRegistry {
    /// Build the default format registry with all known conversion paths.
    fn new() -> Self {
        let mut conversions: HashMap<Format, Vec<(Format, bool, String)>> = HashMap::new();

        // APR can convert to/from everything (hub format)
        conversions.entry(Format::Apr).or_default().extend(vec![
            (
                Format::SafeTensors,
                false,
                "APR → SafeTensors (lossless, header rewrite)".into(),
            ),
            (
                Format::Gguf,
                false,
                "APR → GGUF (lossless, metadata mapping)".into(),
            ),
            (
                Format::Onnx,
                true,
                "APR → ONNX (lossy, graph reconstruction)".into(),
            ),
        ]);

        conversions
            .entry(Format::SafeTensors)
            .or_default()
            .extend(vec![(
                Format::Apr,
                false,
                "SafeTensors → APR (lossless, direct import)".into(),
            )]);

        conversions.entry(Format::Gguf).or_default().extend(vec![(
            Format::Apr,
            false,
            "GGUF → APR (lossless, dequantize if needed)".into(),
        )]);

        conversions.entry(Format::Onnx).or_default().extend(vec![(
            Format::Apr,
            true,
            "ONNX → APR (lossy, weight extraction)".into(),
        )]);

        Self { conversions }
    }

    /// Find the shortest conversion path between two formats.
    ///
    /// Uses BFS to find the optimal path. Returns identity for same-format,
    /// direct path if available, or transitive path through APR hub.
    fn find_path(&self, from: Format, to: Format) -> Option<ConversionPath> {
        // Identity case
        if from == to {
            return Some(ConversionPath {
                from,
                to,
                steps: vec![],
            });
        }

        // BFS for shortest path
        let mut visited = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        visited.insert(from, None);

        while let Some(current) = queue.pop_front() {
            if current == to {
                break;
            }

            if let Some(neighbors) = self.conversions.get(&current) {
                for (next, lossy, desc) in neighbors {
                    if !visited.contains_key(next) {
                        visited.insert(
                            *next,
                            Some(ConversionStep {
                                from: current,
                                to: *next,
                                lossy: *lossy,
                                description: desc.clone(),
                            }),
                        );
                        queue.push_back(*next);
                    }
                }
            }
        }

        // Reconstruct path
        if !visited.contains_key(&to) {
            return None;
        }

        let mut steps = Vec::new();
        let mut current = to;
        while let Some(Some(step)) = visited.get(&current) {
            steps.push(step.clone());
            current = step.from;
        }
        steps.reverse();

        Some(ConversionPath { from, to, steps })
    }
}

// ---------------------------------------------------------------------------
// Conversion execution
// ---------------------------------------------------------------------------

/// Simulated model data for conversion.
struct ModelData {
    format: Format,
    payload: Vec<u8>,
    tensor_count: usize,
}

/// Execute a conversion step by transforming model data.
fn execute_step(data: &ModelData, step: &ConversionStep) -> ModelData {
    assert_eq!(data.format, step.from);

    // Simulate format conversion — the payload changes slightly
    // based on the target format (header differences, alignment, etc.)
    let mut new_payload = Vec::new();

    // Write format-specific header
    match step.to {
        Format::Apr => {
            new_payload.extend_from_slice(b"APR2");
            new_payload.extend_from_slice(&data.payload);
        }
        Format::SafeTensors => {
            // SafeTensors: 8-byte header length + JSON header + data
            let header = b"{}";
            new_payload.extend_from_slice(&(header.len() as u64).to_le_bytes());
            new_payload.extend_from_slice(header);
            new_payload.extend_from_slice(&data.payload);
        }
        Format::Gguf => {
            new_payload.extend_from_slice(b"GGUF");
            new_payload.extend_from_slice(&3u32.to_le_bytes());
            new_payload.extend_from_slice(&data.payload);
        }
        Format::Onnx => {
            // ONNX protobuf-style header
            new_payload.extend_from_slice(b"\x08\x07"); // ONNX IR version 7
            new_payload.extend_from_slice(&data.payload);
        }
    }

    ModelData {
        format: step.to,
        payload: new_payload,
        tensor_count: data.tensor_count,
    }
}

/// Execute a full conversion path.
fn convert(data: ModelData, path: &ConversionPath) -> ModelData {
    let mut current = data;
    for step in &path.steps {
        current = execute_step(&current, step);
    }
    current
}

// ---------------------------------------------------------------------------
// Section helpers
// ---------------------------------------------------------------------------

/// Section 1: Display the format registry — all formats and their direct targets.
fn print_format_registry(registry: &FormatRegistry) {
    println!("=== Format Registry ===");
    let all_formats = [Format::Apr, Format::SafeTensors, Format::Gguf, Format::Onnx];
    for fmt in &all_formats {
        let targets: Vec<String> = registry
            .conversions
            .get(fmt)
            .map(|v| v.iter().map(|(t, _, _)| format!("{t}")).collect())
            .unwrap_or_default();
        println!("  {fmt} → [{}]", targets.join(", "));
    }
    println!();
}

/// Classify a conversion path as identity, direct, or transitive.
fn path_kind(path: &ConversionPath) -> &'static str {
    if path.is_identity() {
        "identity"
    } else if path.is_direct() {
        "direct"
    } else {
        "transitive"
    }
}

/// Section 2: Find and display conversion paths for representative format pairs.
fn print_conversion_paths(registry: &FormatRegistry) {
    println!("=== Conversion Paths ===");
    let test_pairs = [
        (Format::SafeTensors, Format::Apr),
        (Format::SafeTensors, Format::Gguf),
        (Format::Gguf, Format::Onnx),
        (Format::Apr, Format::Apr),
    ];

    for (from, to) in &test_pairs {
        match registry.find_path(*from, *to) {
            Some(path) => {
                let kind = path_kind(&path);
                let lossy = if path.is_lossy() { " (lossy)" } else { "" };
                println!(
                    "  {from} → {to}: {kind}{lossy}, {} step(s)",
                    path.steps.len()
                );
                for (i, step) in path.steps.iter().enumerate() {
                    println!("    Step {}: {}", i + 1, step.description);
                }
            }
            None => println!("  {from} → {to}: NO PATH"),
        }
    }
    println!();
}

/// Section 3: Execute a direct SafeTensors → APR conversion and print results.
fn run_direct_conversion(registry: &FormatRegistry) {
    println!("=== Conversion Execution: SafeTensors → APR ===");
    let source = ModelData {
        format: Format::SafeTensors,
        payload: generate_model_payload(42, 4096),
        tensor_count: 4,
    };
    println!(
        "Source: {} ({} bytes, {} tensors)",
        source.format,
        source.payload.len(),
        source.tensor_count
    );

    let path = registry
        .find_path(Format::SafeTensors, Format::Apr)
        .unwrap();
    let result = convert(source, &path);
    println!(
        "Result: {} ({} bytes, {} tensors)",
        result.format,
        result.payload.len(),
        result.tensor_count
    );
    assert_eq!(result.format, Format::Apr);
    println!();
}

/// Section 4: Execute a transitive SafeTensors → GGUF conversion (via APR hub).
fn run_transitive_conversion(registry: &FormatRegistry) {
    println!("=== Transitive Conversion: SafeTensors → GGUF ===");
    let source = ModelData {
        format: Format::SafeTensors,
        payload: generate_model_payload(43, 2048),
        tensor_count: 3,
    };
    let path = registry
        .find_path(Format::SafeTensors, Format::Gguf)
        .unwrap();
    println!("Steps: {}", path.steps.len());
    for step in &path.steps {
        println!("  {} → {}", step.from, step.to);
    }
    let result = convert(source, &path);
    assert_eq!(result.format, Format::Gguf);
    println!(
        "Output format: {} ({} bytes)",
        result.format,
        result.payload.len()
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_rosetta_convert")?;

    let registry = FormatRegistry::new();

    print_format_registry(&registry);
    print_conversion_paths(&registry);
    run_direct_conversion(&registry);
    run_transitive_conversion(&registry);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_format_is_identity() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Apr, Format::Apr).unwrap();
        assert!(path.is_identity());
        assert_eq!(path.steps.len(), 0);
    }

    #[test]
    fn test_direct_conversion_safetensors_to_apr() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        assert!(path.is_direct());
        assert_eq!(path.steps.len(), 1);
    }

    #[test]
    fn test_transitive_conversion_safetensors_to_gguf() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Gguf).unwrap();
        assert_eq!(path.steps.len(), 2); // ST → APR → GGUF
        assert_eq!(path.steps[0].from, Format::SafeTensors);
        assert_eq!(path.steps[0].to, Format::Apr);
        assert_eq!(path.steps[1].from, Format::Apr);
        assert_eq!(path.steps[1].to, Format::Gguf);
    }

    #[test]
    fn test_transitive_conversion_gguf_to_onnx() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Gguf, Format::Onnx).unwrap();
        assert_eq!(path.steps.len(), 2); // GGUF → APR → ONNX
    }

    #[test]
    fn test_lossy_path_detected() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Apr, Format::Onnx).unwrap();
        assert!(path.is_lossy());
    }

    #[test]
    fn test_lossless_path() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        assert!(!path.is_lossy());
    }

    #[test]
    fn test_execute_conversion() {
        let reg = FormatRegistry::new();
        let data = ModelData {
            format: Format::SafeTensors,
            payload: generate_model_payload(42, 1024),
            tensor_count: 2,
        };
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        let result = convert(data, &path);
        assert_eq!(result.format, Format::Apr);
        assert_eq!(result.tensor_count, 2);
        assert!(&result.payload[0..4] == b"APR2");
    }

    #[test]
    fn test_identity_conversion_preserves_data() {
        let reg = FormatRegistry::new();
        let payload = generate_model_payload(42, 512);
        let data = ModelData {
            format: Format::Apr,
            payload: payload.clone(),
            tensor_count: 1,
        };
        let path = reg.find_path(Format::Apr, Format::Apr).unwrap();
        let result = convert(data, &path);
        assert_eq!(result.payload, payload);
    }

    #[test]
    fn test_all_formats_reachable_from_apr() {
        let reg = FormatRegistry::new();
        for fmt in [Format::SafeTensors, Format::Gguf, Format::Onnx] {
            assert!(
                reg.find_path(Format::Apr, fmt).is_some(),
                "APR → {fmt} should have a path"
            );
        }
    }

    #[test]
    fn test_all_formats_can_reach_apr() {
        let reg = FormatRegistry::new();
        for fmt in [Format::SafeTensors, Format::Gguf, Format::Onnx] {
            assert!(
                reg.find_path(fmt, Format::Apr).is_some(),
                "{fmt} → APR should have a path"
            );
        }
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", Format::Apr), "APR");
        assert_eq!(format!("{}", Format::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", Format::Gguf), "GGUF");
        assert_eq!(format!("{}", Format::Onnx), "ONNX");
    }
}
