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

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args)?;

    if config.help {
        print_help();
        return Ok(());
    }

    run_convert(&config)
}

#[derive(Debug, Clone)]
struct ConvertConfig {
    input_path: Option<String>,
    output_path: Option<String>,
    output_format: OutputFormat,
    quantize: Option<String>,
    demo: bool,
    verbose: bool,
    help: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Apr,
    Gguf,
    SafeTensors,
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

fn parse_args(args: &[String]) -> Result<ConvertConfig> {
    let mut config = ConvertConfig {
        input_path: None,
        output_path: None,
        output_format: OutputFormat::Apr,
        quantize: None,
        demo: false,
        verbose: false,
        help: false,
    };

    let mut positional = 0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => config.help = true,
            "--demo" | "-d" => config.demo = true,
            "--verbose" | "-v" => config.verbose = true,
            "--format" | "-f" => {
                i += 1;
                if i < args.len() {
                    config.output_format = match args[i].as_str() {
                        "apr" => OutputFormat::Apr,
                        "gguf" => OutputFormat::Gguf,
                        "safetensors" | "st" => OutputFormat::SafeTensors,
                        _ => OutputFormat::Apr,
                    };
                }
            }
            "--quantize" | "-q" => {
                i += 1;
                if i < args.len() {
                    config.quantize = Some(args[i].clone());
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    config.output_path = Some(args[i].clone());
                }
            }
            path if !path.starts_with('-') => {
                if positional == 0 {
                    config.input_path = Some(path.to_string());
                    positional += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("apr-convert - Convert between model formats");
    println!();
    println!("USAGE:");
    println!("    apr-convert [OPTIONS] <INPUT>");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help             Print help information");
    println!("    -d, --demo             Run with demo model");
    println!("    -v, --verbose          Verbose output");
    println!("    -f, --format FORMAT    Output format (apr, gguf, safetensors)");
    println!("    -o, --output PATH      Output file path");
    println!("    -q, --quantize LEVEL   Quantization (q4_0, q8_0, fp16)");
    println!();
    println!("SUPPORTED FORMATS:");
    println!("    apr         - APR native format");
    println!("    gguf        - GGML Universal Format");
    println!("    safetensors - HuggingFace SafeTensors");
    println!();
    println!("EXAMPLES:");
    println!("    apr-convert model.safetensors -f apr");
    println!("    apr-convert model.apr -f gguf -q q4_0");
    println!("    apr-convert --demo -f gguf");
}

fn run_convert(config: &ConvertConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_convert")?;

    // Get input
    let (input_path, input_bytes) = if config.demo {
        let payload = generate_model_payload(42, 2048);
        let bytes = ModelBundle::new()
            .with_name("demo")
            .with_compression(true)
            .with_payload(payload)
            .build();
        ("demo.apr".to_string(), bytes)
    } else if let Some(path) = &config.input_path {
        let bytes = std::fs::read(path)?;
        (path.clone(), bytes)
    } else {
        print_help();
        return Ok(());
    };

    let input_format = detect_format(&input_bytes);

    if config.verbose {
        println!(
            "Input: {} ({}, {} bytes)",
            input_path,
            input_format,
            input_bytes.len()
        );
    }

    // Convert
    let output_bytes = convert(
        &input_bytes,
        config.output_format,
        config.quantize.as_deref(),
    )?;

    let output_format_str = match config.output_format {
        OutputFormat::Apr => "apr",
        OutputFormat::Gguf => "gguf",
        OutputFormat::SafeTensors => "safetensors",
    };

    // Determine output path
    let output_path = config.output_path.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&input_path)
            .file_stem().map_or_else(|| "output".to_string(), |s| s.to_string_lossy().to_string());

        let ext = match config.output_format {
            OutputFormat::Apr => "apr",
            OutputFormat::Gguf => "gguf",
            OutputFormat::SafeTensors => "safetensors",
        };

        format!("{}.{}", stem, ext)
    });

    // Write output (in demo mode, write to temp dir)
    let actual_output_path = if config.demo {
        let temp_path = ctx.path(&output_path);
        std::fs::write(&temp_path, &output_bytes)?;
        temp_path.to_string_lossy().to_string()
    } else {
        std::fs::write(&output_path, &output_bytes)?;
        output_path.clone()
    };

    let compression_ratio = input_bytes.len() as f64 / output_bytes.len() as f64;

    ctx.record_metric("input_size", input_bytes.len() as i64);
    ctx.record_metric("output_size", output_bytes.len() as i64);
    ctx.record_float_metric("compression_ratio", compression_ratio);

    // Print result
    println!("Conversion complete!");
    println!();
    println!("Input:  {} ({})", input_path, input_format);
    println!("Output: {} ({})", actual_output_path, output_format_str);
    println!();
    println!("Input size:  {} bytes", input_bytes.len());
    println!("Output size: {} bytes", output_bytes.len());
    println!("Ratio: {:.2}x", compression_ratio);

    if config.quantize.is_some() {
        println!(
            "Quantization: {}",
            config.quantize.as_ref().unwrap_or(&"none".to_string())
        );
    }

    Ok(())
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

fn convert(
    input: &[u8],
    output_format: OutputFormat,
    quantize: Option<&str>,
) -> Result<Vec<u8>> {
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

        assert_eq!(config.output_format, OutputFormat::Gguf);
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
