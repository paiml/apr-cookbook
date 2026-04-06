//! # Recipe: APR Format Converter CLI
//!
//! **Category**: CLI Tools
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Convert between model formats from command line.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_convert
//! cargo run --example cli_apr_convert -- --demo
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let config = ConvertConfig::parse();
    run_convert(&config)
}

#[derive(Debug, Clone, Parser)]
#[command(name = "apr-convert", about = "Convert between model formats")]
struct ConvertConfig {
    /// Input model file path
    input_path: Option<String>,

    /// Output file path
    #[arg(short = 'o', long = "output")]
    output_path: Option<String>,

    /// Output format (apr, gguf, safetensors)
    #[arg(short = 'f', long = "format", default_value = "apr")]
    output_format_str: String,

    /// Quantization level (q4_0, q8_0, fp16)
    #[arg(short, long)]
    quantize: Option<String>,

    /// Run with demo model
    #[arg(long, short = 'd')]
    demo: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

impl ConvertConfig {
    fn output_format(&self) -> OutputFormat {
        parse_output_format(&self.output_format_str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Apr,
    Gguf,
    SafeTensors,
}

impl OutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Apr => "apr",
            Self::Gguf => "gguf",
            Self::SafeTensors => "safetensors",
        }
    }

    fn extension(self) -> &'static str {
        self.as_str()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct ConversionResult {
    input_path: String,
    output_path: String,
    input_format: String,
    output_format: String,
    input_size: usize,
    output_size: usize,
    compression_ratio: f64,
    quantized: bool,
}

/// Parse output format from string
fn parse_output_format(s: &str) -> OutputFormat {
    match s {
        "gguf" => OutputFormat::Gguf,
        "safetensors" | "st" => OutputFormat::SafeTensors,
        _ => OutputFormat::Apr,
    }
}

#[cfg(test)]
fn parse_args(args: &[String]) -> std::result::Result<ConvertConfig, clap::Error> {
    ConvertConfig::try_parse_from(args)
}

/// Load input bytes from config (demo mode or file)
fn load_input(config: &ConvertConfig) -> Option<(String, Vec<u8>)> {
    if config.demo {
        let payload = generate_model_payload(42, 2048);
        let bytes = ModelBundle::new()
            .with_name("demo")
            .with_compression(true)
            .with_payload(payload)
            .build();
        Some(("demo.apr".to_string(), bytes))
    } else {
        config
            .input_path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok().map(|bytes| (path.clone(), bytes)))
    }
}

/// Generate output path from input path and format
fn generate_output_path(input_path: &str, format: OutputFormat) -> String {
    let stem = std::path::Path::new(input_path)
        .file_stem()
        .map_or_else(|| "output".to_string(), |s| s.to_string_lossy().to_string());
    format!("{}.{}", stem, format.extension())
}

/// Write output and return the actual path written
fn write_output(
    ctx: &mut RecipeContext,
    output_path: &str,
    output_bytes: &[u8],
    demo: bool,
) -> Result<String> {
    if demo {
        let temp_path = ctx.path(output_path);
        std::fs::write(&temp_path, output_bytes)?;
        Ok(temp_path.to_string_lossy().to_string())
    } else {
        std::fs::write(output_path, output_bytes)?;
        Ok(output_path.to_string())
    }
}

fn run_convert(config: &ConvertConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_convert")?;
    let output = AdaptiveOutput::new();

    // Phase 1: Load input
    output.progress(1, 4, "loading model");
    let Some((input_path, input_bytes)) = load_input(config) else {
        println!("No input provided. Use --demo or specify an input file.");
        return Ok(());
    };

    // Phase 2: Detect format
    output.progress(2, 4, "detecting format");
    let input_format = detect_format(&input_bytes);

    if config.verbose {
        println!(
            "Input: {} ({}, {} bytes)",
            input_path,
            input_format,
            input_bytes.len()
        );
    }

    // Phase 3: Convert
    output.progress(
        3,
        4,
        &format!("converting to {}", config.output_format().as_str()),
    );
    let output_bytes = convert(
        &input_bytes,
        config.output_format(),
        config.quantize.as_deref(),
    )?;

    // Phase 4: Write output
    output.progress(4, 4, "writing output");
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| generate_output_path(&input_path, config.output_format()));
    let actual_output_path = write_output(&mut ctx, &output_path, &output_bytes, config.demo)?;
    output.status(""); // clear progress line

    // Record metrics
    let compression_ratio = input_bytes.len() as f64 / output_bytes.len() as f64;
    ctx.record_metric("input_size", input_bytes.len() as i64);
    ctx.record_metric("output_size", output_bytes.len() as i64);
    ctx.record_float_metric("compression_ratio", compression_ratio);

    // Print result
    print_result(
        &input_path,
        &input_format,
        &actual_output_path,
        config,
        &input_bytes,
        &output_bytes,
        compression_ratio,
    );

    Ok(())
}

fn print_result(
    input_path: &str,
    input_format: &str,
    output_path: &str,
    config: &ConvertConfig,
    input_bytes: &[u8],
    output_bytes: &[u8],
    compression_ratio: f64,
) {
    println!("Conversion complete!");
    println!();
    println!("Input:  {} ({})", input_path, input_format);
    println!(
        "Output: {} ({})",
        output_path,
        config.output_format().as_str()
    );
    println!();
    println!("Input size:  {} bytes", input_bytes.len());
    println!("Output size: {} bytes", output_bytes.len());
    println!("Ratio: {:.2}x", compression_ratio);

    if let Some(q) = &config.quantize {
        println!("Quantization: {}", q);
    }
}

fn detect_format(bytes: &[u8]) -> String {
    if bytes.len() >= 4 {
        let magic = &bytes[0..4];
        if magic == b"APRN" {
            return "apr".to_string();
        } else if magic == b"GGUF" {
            return "gguf".to_string();
        } else if bytes.len() >= 8 && &bytes[0..8] == b"{\"metada" {
            return "safetensors".to_string();
        }
    }
    "unknown".to_string()
}

fn convert(input: &[u8], output_format: OutputFormat, quantize: Option<&str>) -> Result<Vec<u8>> {
    // Simulated conversion
    let base_output = match output_format {
        OutputFormat::Apr => ModelBundle::new()
            .with_compression(true)
            .with_payload(input.to_vec())
            .build(),
        OutputFormat::Gguf => {
            // Mock GGUF header + data
            let mut output = b"GGUF".to_vec();
            output.extend(input.iter().take(input.len().min(1000)));
            output
        }
        OutputFormat::SafeTensors => {
            // Mock SafeTensors format
            let mut output = b"{\"metadata\":{}}\n".to_vec();
            output.extend(input.iter().take(input.len().min(1000)));
            output
        }
    };

    // Apply quantization simulation
    let output = if let Some(q) = quantize {
        let factor = match q {
            "q4_0" => 0.25,
            "q8_0" => 0.5,
            "fp16" => 0.5,
            _ => 1.0,
        };
        base_output
            .iter()
            .take((base_output.len() as f64 * factor) as usize)
            .copied()
            .collect()
    } else {
        base_output
    };

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-convert".to_string(), "--demo".to_string()];
        let config = parse_args(&args).unwrap();

        assert!(config.demo);
    }

    #[test]
    fn test_parse_args_format() {
        let args = vec![
            "apr-convert".to_string(),
            "-f".to_string(),
            "gguf".to_string(),
        ];
        let config = parse_args(&args).unwrap();

        assert_eq!(config.output_format(), OutputFormat::Gguf);
    }

    #[test]
    fn test_parse_args_quantize() {
        let args = vec![
            "apr-convert".to_string(),
            "-q".to_string(),
            "q4_0".to_string(),
        ];
        let config = parse_args(&args).unwrap();

        assert_eq!(config.quantize, Some("q4_0".to_string()));
    }

    #[test]
    fn test_detect_format_apr() {
        let bytes = b"APRN\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), "apr");
    }

    #[test]
    fn test_detect_format_gguf() {
        let bytes = b"GGUF\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), "gguf");
    }

    #[test]
    fn test_convert_to_apr() {
        let input = vec![1, 2, 3, 4, 5];
        let output = convert(&input, OutputFormat::Apr, None).unwrap();

        assert!(!output.is_empty());
    }

    #[test]
    fn test_convert_to_gguf() {
        let input = vec![1, 2, 3, 4, 5];
        let output = convert(&input, OutputFormat::Gguf, None).unwrap();

        assert!(&output[0..4] == b"GGUF");
    }

    #[test]
    fn test_quantize_reduces_size() {
        let input = vec![0u8; 1000];
        let output_full = convert(&input, OutputFormat::Apr, None).unwrap();
        let output_q4 = convert(&input, OutputFormat::Apr, Some("q4_0")).unwrap();

        assert!(output_q4.len() < output_full.len());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_convert_produces_output(input in proptest::collection::vec(0u8..255, 10..100)) {
            let output = convert(&input, OutputFormat::Apr, None).unwrap();
            prop_assert!(!output.is_empty());
        }

        #[test]
        fn prop_quantize_reduces_size(input in proptest::collection::vec(0u8..255, 100..500)) {
            let full = convert(&input, OutputFormat::Apr, None).unwrap();
            let q4 = convert(&input, OutputFormat::Apr, Some("q4_0")).unwrap();

            prop_assert!(q4.len() <= full.len());
        }
    }
}
