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
// Format types
// ---------------------------------------------------------------------------

/// Supported model formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
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
pub struct ConversionStep {
    pub from: Format,
    pub to: Format,
    pub lossy: bool,
    pub description: String,
}

/// A complete conversion path from source to target format.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConversionPath {
    pub from: Format,
    pub to: Format,
    pub steps: Vec<ConversionStep>,
}

impl ConversionPath {
    pub fn is_direct(&self) -> bool {
        self.steps.len() == 1
    }

    pub fn is_identity(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn is_lossy(&self) -> bool {
        self.steps.iter().any(|s| s.lossy)
    }
}

/// Format registry with conversion capabilities.
pub struct FormatRegistry {
    // Adjacency list: from → [(to, lossy, description)]
    pub conversions: HashMap<Format, Vec<(Format, bool, String)>>,
}

// ---------------------------------------------------------------------------
// Registry and path finding
// ---------------------------------------------------------------------------

impl FormatRegistry {
    /// Build the default format registry with all known conversion paths.
    pub fn new() -> Self {
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

    // Find the shortest conversion path between two formats.
    //
    // Uses BFS to find the optimal path. Returns identity for same-format,
    /// direct path if available, or transitive path through APR hub.
    pub fn find_path(&self, from: Format, to: Format) -> Option<ConversionPath> {
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
pub struct ModelData {
    pub format: Format,
    pub payload: Vec<u8>,
    pub tensor_count: usize,
}

/// Execute a conversion step by transforming model data.
pub fn execute_step(data: &ModelData, step: &ConversionStep) -> ModelData {
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
pub fn convert(data: ModelData, path: &ConversionPath) -> ModelData {
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
pub fn print_format_registry(registry: &FormatRegistry) {
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
pub fn path_kind(path: &ConversionPath) -> &'static str {
    if path.is_identity() {
        "identity"
    } else if path.is_direct() {
        "direct"
    } else {
        "transitive"
    }
}

/// Section 2: Find and display conversion paths for representative format pairs.
pub fn print_conversion_paths(registry: &FormatRegistry) {
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
pub fn run_direct_conversion(registry: &FormatRegistry) {
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
pub fn run_transitive_conversion(registry: &FormatRegistry) {
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
