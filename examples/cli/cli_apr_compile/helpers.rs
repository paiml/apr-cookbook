#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CompileConfig {
    pub model_path: Option<String>,
    pub output_path: Option<String>,
    pub target: Option<String>,
    pub release: bool,
    pub strip: bool,
    pub lto: bool,
    pub list_targets: bool,
    pub demo: bool,
    pub verbose: bool,
    pub help: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileReport {
    pub model_name: String,
    pub model_size: usize,
    pub architecture: String,
    pub target: String,
    pub release: bool,
    pub strip: bool,
    pub lto: bool,
    pub cargo_toml: String,
    pub main_rs_lines: usize,
    pub estimated_binary_size: String,
}

/// Known compilation targets (APR-SPEC §4.16.4).
pub const TARGETS: &[(&str, &str)] = &[
    ("x86_64-unknown-linux-gnu", "Linux x86_64 (glibc)"),
    (
        "x86_64-unknown-linux-musl",
        "Linux x86_64 (musl, fully static)",
    ),
    ("aarch64-unknown-linux-gnu", "Linux ARM64"),
    ("x86_64-apple-darwin", "macOS x86_64"),
    ("aarch64-apple-darwin", "macOS ARM64 (Apple Silicon)"),
    ("x86_64-pc-windows-msvc", "Windows x86_64"),
    ("wasm32-unknown-unknown", "Pure WASM (browser)"),
    ("wasm32-wasi", "WASM + WASI (server-side)"),
    ("wasm32-wasip1", "WASM + WASI Preview 1"),
    ("wasm32-wasip2", "WASM + WASI Preview 2 (component model)"),
];

pub fn parse_args(args: &[String]) -> Result<CompileConfig> {
    let mut config = CompileConfig {
        model_path: None,
        output_path: None,
        target: None,
        release: false,
        strip: false,
        lto: false,
        list_targets: false,
        demo: false,
        verbose: false,
        help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => config.help = true,
            "--demo" | "-d" => config.demo = true,
            "--verbose" | "-v" => config.verbose = true,
            "--release" => config.release = true,
            "--strip" => config.strip = true,
            "--lto" => config.lto = true,
            "--list-targets" => config.list_targets = true,
            "--target" => {
                i += 1;
                if i < args.len() {
                    config.target = Some(args[i].clone());
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    config.output_path = Some(args[i].clone());
                }
            }
            path if !path.starts_with('-') => {
                config.model_path = Some(path.to_string());
            }
            _ => {
                return Err(CookbookError::invalid_format(format!(
                    "Unknown argument: {}",
                    args[i]
                )));
            }
        }
        i += 1;
    }

    Ok(config)
}

pub fn print_help() {
    println!("apr-compile - Compile .apr model into standalone executable (APR-SPEC §4.16)");
    println!();
    println!("USAGE:");
    println!("    apr-compile [OPTIONS] <MODEL.apr>");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help           Print help information");
    println!("    -d, --demo           Run with demo model");
    println!("    -v, --verbose        Verbose output");
    println!("    -o, --output PATH    Output binary path");
    println!("    --target TRIPLE      Target triple (e.g. x86_64-unknown-linux-musl)");
    println!("    --release            Release mode (optimized)");
    println!("    --strip              Strip debug symbols");
    println!("    --lto                Enable LTO");
    println!("    --list-targets       List available targets");
    println!();
    println!("EXAMPLES:");
    println!("    apr-compile model.apr -o model-bin --release --strip");
    println!("    apr-compile --list-targets");
    println!("    apr-compile --demo");
}

pub fn run_compile(config: &CompileConfig) -> Result<()> {
    if config.list_targets {
        println!("Available Compilation Targets (APR-SPEC §4.16.4)");
        println!("=================================================");
        println!();
        println!("Native:");
        for (triple, desc) in &TARGETS[..6] {
            println!("  {:<40} {}", triple, desc);
        }
        println!();
        println!("WebAssembly:");
        for (triple, desc) in &TARGETS[6..] {
            println!("  {:<40} {}", triple, desc);
        }
        return Ok(());
    }

    let mut ctx = RecipeContext::new("cli_apr_compile")?;
    let output = AdaptiveOutput::new();

    // Create or load model
    output.progress(1, 5, "loading model");
    let (model_name, model_bytes) = if config.demo {
        let payload = generate_model_payload(42, 4096);
        let bytes = ModelBundle::new()
            .with_name("whisper-tiny-demo")
            .with_compression(true)
            .with_payload(payload)
            .build();
        ("whisper-tiny-demo".to_string(), bytes)
    } else if let Some(path) = &config.model_path {
        let bytes = std::fs::read(path).map_err(|e| {
            CookbookError::invalid_format(format!("Failed to read {}: {}", path, e))
        })?;
        let name = Path::new(path)
            .file_stem()
            .map_or("model".to_string(), |s| s.to_string_lossy().to_string());
        (name, bytes)
    } else {
        print_help();
        return Ok(());
    };

    let target = config
        .target
        .clone()
        .unwrap_or_else(|| current_target().to_string());
    let bin_name = model_name.to_lowercase().replace(['.', ' ', '-'], "_");
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| bin_name.clone());

    println!("APR Compile Pipeline (APR-SPEC §4.16)");
    println!("======================================");
    println!();
    println!("Model:        {}", model_name);
    println!(
        "Model size:   {} bytes ({:.2} KB)",
        model_bytes.len(),
        model_bytes.len() as f64 / 1024.0
    );
    println!("Target:       {}", target);
    println!("Output:       {}", output_path);
    println!("Release:      {}", config.release);
    println!("Strip:        {}", config.strip);
    println!("LTO:          {}", config.lto);
    println!();

    // Generate Cargo project (demonstrate the template, don't actually build)
    output.progress(2, 5, "generating Cargo.toml");
    let cargo_toml = generate_cargo_toml(&bin_name);
    output.progress(3, 5, "generating main.rs");
    let main_rs = generate_main_rs(&bin_name, &model_name, model_bytes.len());

    println!("Generated Cargo.toml:");
    println!("---------------------");
    println!("{}", cargo_toml);
    println!();
    println!("Generated src/main.rs ({} lines):", main_rs.lines().count());
    println!("----------------------------------");
    println!("{}", main_rs);

    // Write generated files to temp dir for verification
    output.progress(4, 5, "writing project files");
    let project_dir = ctx.path(&bin_name);
    std::fs::create_dir_all(project_dir.join("src"))?;
    std::fs::write(project_dir.join("Cargo.toml"), &cargo_toml)?;
    std::fs::write(project_dir.join("src/main.rs"), &main_rs)?;
    std::fs::write(project_dir.join("model.apr"), &model_bytes)?;

    // Build RUSTFLAGS
    output.progress(5, 5, "generating report");
    let mut rustflags = Vec::new();
    if config.strip {
        rustflags.push("-C strip=symbols");
    }
    if config.lto {
        rustflags.push("-C lto=fat");
    }

    let estimated_size = estimate_binary_size(model_bytes.len(), config.release, config.strip);

    ctx.record_metric("model_size", model_bytes.len() as i64);
    ctx.record_metric("main_rs_lines", main_rs.lines().count() as i64);

    let report = CompileReport {
        model_name,
        model_size: model_bytes.len(),
        architecture: "auto-detect".to_string(),
        target,
        release: config.release,
        strip: config.strip,
        lto: config.lto,
        cargo_toml,
        main_rs_lines: main_rs.lines().count(),
        estimated_binary_size: estimated_size.clone(),
    };

    println!();
    println!("Compile Report");
    println!("==============");
    println!("Project dir:          {}", project_dir.display());
    println!("Estimated binary:     {}", estimated_size);
    let rustflags_str = rustflags.join(" ");
    println!(
        "RUSTFLAGS:            {}",
        if rustflags.is_empty() {
            "(none)"
        } else {
            &rustflags_str
        }
    );
    println!();
    println!("To actually build:");
    println!(
        "  cd {} && cargo build{}",
        project_dir.display(),
        if config.release { " --release" } else { "" }
    );

    if config.verbose {
        println!();
        println!("Full report JSON:");
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        println!("{}", json);
    }

    Ok(())
}

pub fn generate_cargo_toml(bin_name: &str) -> String {
    format!(
        r#"[package]
name = "{bin_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = {{ version = "4", features = ["derive"] }}

[profile.release]
opt-level = "s"
codegen-units = 1
"#
    )
}

pub fn generate_main_rs(bin_name: &str, model_name: &str, _model_size: usize) -> String {
    format!(
        r#"//! Auto-generated by `apr compile` (APR-SPEC §4.16)

use clap::Parser;

/// Embedded model data
pub const MODEL_DATA: &[u8] = include_bytes!("../model.apr");

#[derive(Parser)]
#[command(name = "{bin_name}")]
#[command(about = "Standalone ML model binary")]
pub struct Cli {{
    /// Input file
    #[arg(value_name = "INPUT")]
    input: Option<String>,

    /// Text prompt
    #[arg(short, long)]
    prompt: Option<String>,

    /// Show model info
    #[arg(long)]
    info: bool,
}}

pub fn main() {{
    let cli = Cli::parse();

    if cli.info {{
        println!("Model: {model_name}");
        println!("Embedded size: {{}} bytes", MODEL_DATA.len());
        return;
    }}

    eprintln!("Model loaded: {{}} bytes", MODEL_DATA.len());
    // Inference dispatch would go here
}}
"#
    )
}

pub fn estimate_binary_size(model_size: usize, release: bool, strip: bool) -> String {
    // Rough estimates based on observed behavior
    let overhead = if release {
        if strip {
            700_000
        } else {
            2_000_000
        }
    } else {
        12_000_000
    };
    let total = model_size + overhead;

    if total >= 1_000_000 {
        format!("{:.1} MB", total as f64 / 1_000_000.0)
    } else {
        format!("{:.1} KB", total as f64 / 1_000.0)
    }
}

pub fn current_target() -> &'static str {
    if cfg!(target_os = "linux") {
        if cfg!(target_arch = "x86_64") {
            "x86_64-unknown-linux-gnu"
        } else {
            "aarch64-unknown-linux-gnu"
        }
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "aarch64-apple-darwin"
        } else {
            "x86_64-apple-darwin"
        }
    } else {
        "x86_64-unknown-linux-gnu"
    }
}
