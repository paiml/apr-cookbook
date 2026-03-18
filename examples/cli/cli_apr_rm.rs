//! # Recipe: APR Model Remove CLI
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
//! Demonstrate `apr rm` — remove a model from the local cache.
//! In demo mode: creates a tempdir cache, populates fake entries,
//! removes one, and verifies cleanup.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_rm
//! cargo run --example cli_apr_rm -- --demo
//! cargo run --example cli_apr_rm -- --demo --dry-run
//! ```

use apr_cookbook::prelude::*;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let config = RmConfig::parse();

    run_rm(&config)
}

/// Remove a cached APR model
#[derive(Debug, Clone, Parser)]
#[command(name = "apr-rm", about = "Remove a cached APR model")]
struct RmConfig {
    /// Model name to remove
    #[arg(value_name = "MODEL_NAME")]
    model_name: Option<String>,

    /// Skip confirmation prompt
    #[arg(short, long)]
    force: bool,

    /// Show what would be removed without deleting
    #[arg(short = 'n', long)]
    dry_run: bool,

    /// Run demo with fake cache
    #[arg(long)]
    demo: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry {
    name: String,
    version: String,
    size_bytes: u64,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct RemovalResult {
    model_name: String,
    bytes_freed: u64,
    remaining_count: usize,
    remaining_bytes: u64,
}

fn run_rm(config: &RmConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_rm")?;

    if config.demo {
        return run_demo(config, &mut ctx);
    }

    let Some(model_name) = &config.model_name else {
        println!("No model specified. Use --demo to see a simulated removal.");
        return Ok(());
    };

    // Non-demo: simulate lookup failure
    println!("Model '{}' not found in cache.", model_name);
    println!("Use --demo to see a simulated removal.");
    Ok(())
}

fn run_demo(config: &RmConfig, ctx: &mut RecipeContext) -> Result<()> {
    let cache_dir = ctx.path("cache");
    std::fs::create_dir_all(&cache_dir)?;

    // Populate fake cache
    let entries = create_demo_cache(&cache_dir)?;

    println!("APR Model Cache ({} models)", entries.len());
    println!("================================");
    print_cache_table(&entries);
    println!();

    let total_before: u64 = entries.iter().map(|e| e.size_bytes).sum();
    ctx.record_metric("models_before", entries.len() as i64);
    ctx.record_metric("total_bytes_before", total_before as i64);

    // Pick model to remove (deterministic: always the third)
    let target_name = config.model_name.clone().unwrap_or_else(|| {
        entries
            .get(2)
            .map_or("whisper-tiny".into(), |e| e.name.clone())
    });

    let Some(target) = entries.iter().find(|e| e.name == target_name).cloned() else {
        println!("Model '{}' not found in cache.", target_name);
        return Ok(());
    };

    // Confirmation simulation
    if config.dry_run {
        println!(
            "[DRY RUN] Would remove '{}' ({}).",
            target.name,
            format_size(target.size_bytes)
        );
        println!("[DRY RUN] No files deleted.");
        return Ok(());
    }

    if !config.force {
        println!(
            "Confirm removal of '{}' ({})? [simulated: yes]",
            target.name,
            format_size(target.size_bytes)
        );
    }

    // Remove the entry
    let result = remove_model(&cache_dir, &entries, &target.name)?;

    println!("Removed '{}' successfully.", result.model_name);
    println!("  Freed: {}", format_size(result.bytes_freed));
    println!(
        "  Remaining: {} model(s), {}",
        result.remaining_count,
        format_size(result.remaining_bytes)
    );
    println!();

    // Show updated cache
    let remaining: Vec<_> = entries
        .iter()
        .filter(|e| e.name != target.name)
        .cloned()
        .collect();
    println!("Updated Cache ({} models)", remaining.len());
    println!("================================");
    print_cache_table(&remaining);

    ctx.record_metric("bytes_freed", result.bytes_freed as i64);
    ctx.record_metric("models_after", result.remaining_count as i64);

    Ok(())
}

fn create_demo_cache(cache_dir: &Path) -> Result<Vec<CacheEntry>> {
    let specs: &[(&str, &str)] = &[
        ("phi-3-mini", "3.1.0"),
        ("llama-3.2-1b", "1.0.0"),
        ("whisper-tiny", "2.0.1"),
        ("bert-base", "4.2.0"),
        ("codellama-7b", "1.1.0"),
    ];

    let mut entries = Vec::new();

    for (name, version) in specs {
        let seed = model_seed(name);
        let size_bytes = seed_to_size(seed);
        let model_dir = cache_dir.join(name);
        std::fs::create_dir_all(&model_dir)?;

        // Write a small placeholder file
        let model_file = model_dir.join("model.apr");
        std::fs::write(&model_file, format!("APR_PLACEHOLDER_{name}"))?;

        entries.push(CacheEntry {
            name: (*name).to_string(),
            version: (*version).to_string(),
            size_bytes,
            path: model_dir,
        });
    }

    Ok(entries)
}

fn remove_model(
    cache_dir: &Path,
    entries: &[CacheEntry],
    target_name: &str,
) -> Result<RemovalResult> {
    let target = entries.iter().find(|e| e.name == target_name);
    let Some(target) = target else {
        return Err(CookbookError::invalid_format(format!(
            "Model '{target_name}' not found in cache"
        )));
    };

    let bytes_freed = target.size_bytes;
    let model_dir = cache_dir.join(target_name);
    if model_dir.exists() {
        std::fs::remove_dir_all(&model_dir)?;
    }

    let remaining: Vec<_> = entries.iter().filter(|e| e.name != target_name).collect();
    let remaining_bytes: u64 = remaining.iter().map(|e| e.size_bytes).sum();

    Ok(RemovalResult {
        model_name: target_name.to_string(),
        bytes_freed,
        remaining_count: remaining.len(),
        remaining_bytes,
    })
}

fn model_seed(name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

fn seed_to_size(seed: u64) -> u64 {
    let base = 39_000_000_u64;
    let range = 7_000_000_000_u64;
    base + (seed % range)
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

fn print_cache_table(entries: &[CacheEntry]) {
    println!("{:<20} {:<10} {:>10}", "NAME", "VERSION", "SIZE");
    println!("{}", "-".repeat(42));
    for e in entries {
        println!(
            "{:<20} {:<10} {:>10}",
            e.name,
            e.version,
            format_size(e.size_bytes)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clap_parse_defaults() {
        let config = RmConfig::parse_from(["apr-rm"]);
        assert!(!config.demo);
        assert!(!config.force);
        assert!(!config.dry_run);
        assert!(config.model_name.is_none());
    }

    #[test]
    fn test_clap_parse_demo() {
        let config = RmConfig::parse_from(["apr-rm", "--demo"]);
        assert!(config.demo);
    }

    #[test]
    fn test_clap_parse_force() {
        let config = RmConfig::parse_from(["apr-rm", "-f", "my-model"]);
        assert!(config.force);
        assert_eq!(config.model_name, Some("my-model".to_string()));
    }

    #[test]
    fn test_clap_parse_dry_run() {
        let config = RmConfig::parse_from(["apr-rm", "--dry-run"]);
        assert!(config.dry_run);
    }

    #[test]
    fn test_clap_parse_unknown() {
        let result = RmConfig::try_parse_from(["apr-rm", "--badarg"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_demo_cache() {
        let ctx = RecipeContext::new("test_create_demo_cache").expect("ctx ok");
        let cache_dir = ctx.path("cache");
        std::fs::create_dir_all(&cache_dir).expect("mkdir ok");
        let entries = create_demo_cache(&cache_dir).expect("create ok");
        assert_eq!(entries.len(), 5);
        for entry in &entries {
            assert!(entry.path.exists());
            assert!(entry.size_bytes > 0);
        }
    }

    #[test]
    fn test_remove_model_success() {
        let ctx = RecipeContext::new("test_remove_model_success").expect("ctx ok");
        let cache_dir = ctx.path("cache");
        std::fs::create_dir_all(&cache_dir).expect("mkdir ok");
        let entries = create_demo_cache(&cache_dir).expect("create ok");
        let result = remove_model(&cache_dir, &entries, "whisper-tiny").expect("rm ok");
        assert_eq!(result.model_name, "whisper-tiny");
        assert!(result.bytes_freed > 0);
        assert_eq!(result.remaining_count, 4);
        assert!(!cache_dir.join("whisper-tiny").exists());
    }

    #[test]
    fn test_remove_model_not_found() {
        let ctx = RecipeContext::new("test_remove_model_not_found").expect("ctx ok");
        let cache_dir = ctx.path("cache");
        std::fs::create_dir_all(&cache_dir).expect("mkdir ok");
        let entries = create_demo_cache(&cache_dir).expect("create ok");
        let result = remove_model(&cache_dir, &entries, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1_500), "1.5 KB");
        assert_eq!(format_size(2_500_000), "2.5 MB");
        assert_eq!(format_size(3_500_000_000), "3.5 GB");
    }

    #[test]
    fn test_run_demo_mode() {
        let config = RmConfig {
            model_name: None,
            force: true,
            dry_run: false,
            demo: true,
        };
        assert!(run_rm(&config).is_ok());
    }

    #[test]
    fn test_run_demo_dry_run() {
        let config = RmConfig {
            model_name: None,
            force: false,
            dry_run: true,
            demo: true,
        };
        assert!(run_rm(&config).is_ok());
    }

    #[test]
    fn test_deterministic_cache() {
        let ctx = RecipeContext::new("test_deterministic_cache").expect("ctx ok");
        let dir1 = ctx.path("cache1");
        let dir2 = ctx.path("cache2");
        std::fs::create_dir_all(&dir1).expect("mkdir ok");
        std::fs::create_dir_all(&dir2).expect("mkdir ok");
        let e1 = create_demo_cache(&dir1).expect("create ok");
        let e2 = create_demo_cache(&dir2).expect("create ok");
        for (a, b) in e1.iter().zip(e2.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.size_bytes, b.size_bytes);
        }
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
        fn prop_seed_deterministic(name in "[a-z]{3,20}") {
            let s1 = model_seed(&name);
            let s2 = model_seed(&name);
            prop_assert_eq!(s1, s2);
        }

        #[test]
        fn prop_seed_to_size_in_range(seed in 0u64..u64::MAX) {
            let size = seed_to_size(seed);
            prop_assert!(size >= 39_000_000);
        }
    }
}
