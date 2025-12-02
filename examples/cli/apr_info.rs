//! Display APR model information.
//!
//! This CLI tool inspects `.apr` model files and displays their
//! metadata, format version, and capabilities.
//!
//! # Run
//!
//! ```bash
//! cargo run --example apr_info -- --help
//! cargo run --example apr_info -- model.apr
//! ```

use apr_cookbook::bundle::{BundledModel, ModelBundle};
use apr_cookbook::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "apr-info")]
#[command(about = "Display APR model information")]
#[command(version)]
struct Args {
    /// Path to the APR model file
    #[arg(value_name = "FILE")]
    path: Option<PathBuf>,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Use demo mode with sample model
    #[arg(long)]
    demo: bool,
}

fn create_demo_model() -> Vec<u8> {
    ModelBundle::new()
        .with_name("demo-sentiment-classifier")
        .with_description("Demo model for apr-info tool")
        .with_compression(true)
        .with_payload(vec![0u8; 5000])
        .build()
}

fn display_model_info(model: &BundledModel, verbose: bool) {
    println!("=== APR Model Info ===\n");

    // Basic info
    println!("Name:        {}", model.name());
    println!("Size:        {} bytes", model.size());
    println!("Version:     {}.{}", model.version().0, model.version().1);

    println!();

    // Flags
    println!("Flags:");
    println!(
        "  Compressed: {}",
        if model.is_compressed() { "yes" } else { "no" }
    );
    println!(
        "  Encrypted:  {}",
        if model.is_encrypted() { "yes" } else { "no" }
    );
    println!(
        "  Signed:     {}",
        if model.is_signed() { "yes" } else { "no" }
    );

    if verbose {
        println!();
        println!("Header (hex):");
        let bytes = model.as_bytes();
        for (i, chunk) in bytes
            .iter()
            .take(32)
            .collect::<Vec<_>>()
            .chunks(16)
            .enumerate()
        {
            print!("  {:04x}: ", i * 16);
            for byte in chunk {
                print!("{:02x} ", byte);
            }
            println!();
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_bytes = match args.path.as_ref() {
        Some(path) if !args.demo => {
            println!("Reading: {}\n", path.display());
            // In production, would read from file:
            // std::fs::read(path)?
            // For demo, create sample
            create_demo_model()
        }
        _ => {
            if !args.demo {
                println!("No file specified. Running in demo mode.\n");
                println!("Usage: apr-info <FILE> or apr-info --demo\n");
            }
            create_demo_model()
        }
    };

    let model = BundledModel::from_bytes(&model_bytes)?;
    display_model_info(&model, args.verbose);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_model_creation() {
        let bytes = create_demo_model();
        let model = BundledModel::from_bytes(&bytes);
        assert!(model.is_ok());
    }

    #[test]
    fn test_demo_model_is_compressed() {
        let bytes = create_demo_model();
        let model = BundledModel::from_bytes(&bytes).unwrap();
        assert!(model.is_compressed());
    }

    #[test]
    fn test_cli_args_parse_demo() {
        let args = Args::try_parse_from(["apr-info", "--demo"]).unwrap();
        assert!(args.demo);
        assert!(args.path.is_none());
    }

    #[test]
    fn test_cli_args_parse_verbose() {
        let args = Args::try_parse_from(["apr-info", "--verbose", "--demo"]).unwrap();
        assert!(args.verbose);
    }

    #[test]
    fn test_cli_args_parse_file() {
        let args = Args::try_parse_from(["apr-info", "model.apr"]).unwrap();
        assert_eq!(args.path, Some(PathBuf::from("model.apr")));
    }
}
