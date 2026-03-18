//! # Recipe: APR Model List CLI
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
//! Demonstrate `apr list` — enumerate cached models with Ollama-like UX.
//! Supports table and JSON output formats, deterministic demo data.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_list
//! cargo run --example cli_apr_list -- --demo
//! cargo run --example cli_apr_list -- --demo --json
//! ```

use apr_cookbook::prelude::*;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = ListConfig::parse();

    run_list(&config)
}

/// List cached APR models (Ollama-like UX)
#[derive(Debug, Clone, Parser)]
#[command(name = "apr-list", about = "List cached APR models (Ollama-like UX)")]
struct ListConfig {
    /// Output in JSON format
    #[arg(long)]
    json: bool,

    /// Sort by field: name, size, downloaded, last-used
    #[arg(short, long, value_enum, default_value_t = SortField::Name)]
    sort: SortField,

    /// Show demo cached models
    #[arg(long)]
    demo: bool,
}

impl ListConfig {
    fn format(&self) -> OutputFormat {
        if self.json {
            OutputFormat::Json
        } else {
            OutputFormat::Table
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Table,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SortField {
    Name,
    Size,
    Downloaded,
    #[value(name = "last-used")]
    LastUsed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedModel {
    name: String,
    version: String,
    size_bytes: u64,
    downloaded_at: String,
    last_used: String,
    path: String,
}

fn run_list(config: &ListConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_list")?;

    let mut models = if config.demo {
        generate_demo_models()
    } else {
        // In non-demo mode, scan real cache (simulate empty)
        println!("No cached models found. Use --demo for sample data.");
        return Ok(());
    };

    sort_models(&mut models, config.sort);

    let format = config.format();
    match format {
        OutputFormat::Table => print_table(&models),
        OutputFormat::Json => print_json(&models)?,
    }

    let total_size: u64 = models.iter().map(|m| m.size_bytes).sum();
    ctx.record_metric("model_count", models.len() as i64);
    ctx.record_metric("total_size_bytes", total_size as i64);

    if format == OutputFormat::Table {
        println!();
        println!(
            "Total: {} model(s), {}",
            models.len(),
            format_size(total_size)
        );
    }

    Ok(())
}

fn generate_demo_models() -> Vec<CachedModel> {
    let specs: &[(&str, &str, &str)] = &[
        ("phi-3-mini", "3.1.0", "models/phi-3-mini.apr"),
        ("llama-3.2-1b", "1.0.0", "models/llama-3.2-1b.apr"),
        ("whisper-tiny", "2.0.1", "models/whisper-tiny.apr"),
        ("bert-base", "4.2.0", "models/bert-base.apr"),
        ("codellama-7b", "1.1.0", "models/codellama-7b.apr"),
    ];

    specs
        .iter()
        .map(|(name, version, path)| {
            let seed = deterministic_model_seed(name);
            let size_bytes = seed_to_size(seed);
            CachedModel {
                name: (*name).to_string(),
                version: (*version).to_string(),
                size_bytes,
                downloaded_at: seed_to_date(seed, 0),
                last_used: seed_to_date(seed, 1),
                path: (*path).to_string(),
            }
        })
        .collect()
}

fn deterministic_model_seed(name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

fn seed_to_size(seed: u64) -> u64 {
    // Range: 39 MB to ~7 GB
    let base = 39_000_000_u64;
    let range = 7_000_000_000_u64;
    base + (seed % range)
}

fn seed_to_date(seed: u64, variant: u64) -> String {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    variant.hash(&mut hasher);
    let h = hasher.finish();
    let month = 1 + (h % 12);
    let day = 1 + ((h >> 8) % 28);
    let hour = (h >> 16) % 24;
    let minute = (h >> 24) % 60;
    format!("2026-{month:02}-{day:02} {hour:02}:{minute:02}")
}

fn sort_models(models: &mut [CachedModel], field: SortField) {
    match field {
        SortField::Name => models.sort_by(|a, b| a.name.cmp(&b.name)),
        SortField::Size => models.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes)),
        SortField::Downloaded => models.sort_by(|a, b| a.downloaded_at.cmp(&b.downloaded_at)),
        SortField::LastUsed => models.sort_by(|a, b| b.last_used.cmp(&a.last_used)),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

fn print_table(models: &[CachedModel]) {
    println!(
        "{:<20} {:<10} {:>10} {:<18} {:<18}",
        "NAME", "VERSION", "SIZE", "DOWNLOADED", "LAST USED"
    );
    println!("{}", "-".repeat(78));

    for m in models {
        println!(
            "{:<20} {:<10} {:>10} {:<18} {:<18}",
            m.name,
            m.version,
            format_size(m.size_bytes),
            m.downloaded_at,
            m.last_used
        );
    }
}

fn print_json(models: &[CachedModel]) -> Result<()> {
    let json = serde_json::to_string_pretty(models)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clap_parse_defaults() {
        let config = ListConfig::parse_from(["apr-list"]);
        assert!(!config.demo);
        assert_eq!(config.format(), OutputFormat::Table);
        assert_eq!(config.sort, SortField::Name);
    }

    #[test]
    fn test_clap_parse_demo() {
        let config = ListConfig::parse_from(["apr-list", "--demo"]);
        assert!(config.demo);
    }

    #[test]
    fn test_clap_parse_json() {
        let config = ListConfig::parse_from(["apr-list", "--demo", "--json"]);
        assert_eq!(config.format(), OutputFormat::Json);
    }

    #[test]
    fn test_clap_parse_sort() {
        let config = ListConfig::parse_from(["apr-list", "--sort", "size"]);
        assert_eq!(config.sort, SortField::Size);
    }

    #[test]
    fn test_clap_parse_unknown() {
        let result = ListConfig::try_parse_from(["apr-list", "--badarg"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_demo_models_count() {
        let models = generate_demo_models();
        assert_eq!(models.len(), 5);
    }

    #[test]
    fn test_generate_demo_models_deterministic() {
        let m1 = generate_demo_models();
        let m2 = generate_demo_models();
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.size_bytes, b.size_bytes);
            assert_eq!(a.downloaded_at, b.downloaded_at);
        }
    }

    #[test]
    fn test_sort_by_name() {
        let mut models = generate_demo_models();
        sort_models(&mut models, SortField::Name);
        for w in models.windows(2) {
            assert!(w[0].name <= w[1].name);
        }
    }

    #[test]
    fn test_sort_by_size() {
        let mut models = generate_demo_models();
        sort_models(&mut models, SortField::Size);
        for w in models.windows(2) {
            assert!(w[0].size_bytes >= w[1].size_bytes);
        }
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1_500), "1.5 KB");
        assert_eq!(format_size(2_500_000), "2.5 MB");
        assert_eq!(format_size(3_500_000_000), "3.5 GB");
    }

    #[test]
    fn test_print_json_roundtrip() {
        let models = generate_demo_models();
        let json = serde_json::to_string(&models).expect("serialize ok");
        let parsed: Vec<CachedModel> = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(parsed.len(), models.len());
        assert_eq!(parsed[0].name, models[0].name);
    }

    #[test]
    fn test_seed_to_date_deterministic() {
        let d1 = seed_to_date(42, 0);
        let d2 = seed_to_date(42, 0);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_seed_to_date_different_variants() {
        let d1 = seed_to_date(42, 0);
        let d2 = seed_to_date(42, 1);
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_run_list_demo_table() {
        let config = ListConfig {
            json: false,
            sort: SortField::Name,
            demo: true,
        };
        assert!(run_list(&config).is_ok());
    }

    #[test]
    fn test_run_list_demo_json() {
        let config = ListConfig {
            json: true,
            sort: SortField::Name,
            demo: true,
        };
        assert!(run_list(&config).is_ok());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_format_size_non_empty(bytes in 0u64..10_000_000_000) {
            let formatted = format_size(bytes);
            prop_assert!(!formatted.is_empty());
        }

        #[test]
        fn prop_seed_to_date_valid(seed in 0u64..1_000_000, variant in 0u64..10) {
            let date = seed_to_date(seed, variant);
            prop_assert!(date.starts_with("2026-"));
            prop_assert!(date.len() >= 10);
        }

        #[test]
        fn prop_deterministic_seed_consistent(name in "[a-z]{3,20}") {
            let s1 = deterministic_model_seed(&name);
            let s2 = deterministic_model_seed(&name);
            prop_assert_eq!(s1, s2);
        }
    }
}
