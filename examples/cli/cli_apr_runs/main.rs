#![allow(unused_imports)]
//! # Recipe: APR Training Runs CLI
//!
//! **Category**: CLI Tools
//! **CLI Equivalent**: `apr runs`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
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
//! Demonstrate `apr runs` — list, show, and compare training runs.
//! Generates deterministic training run data with varied hyperparams
//! and provides table, detail, and side-by-side comparison views.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_runs
//! cargo run --example cli_apr_runs -- --demo
//! cargo run --example cli_apr_runs -- --demo show run-001
//! cargo run --example cli_apr_runs -- --demo compare run-001 run-003
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
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = RunsConfig::parse();
    run_runs(&config)
}

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

#[cfg(test)]
fn parse_args(args: &[String]) -> std::result::Result<RunsConfig, clap::Error> {
    RunsConfig::try_parse_from(args)
}

fn run_runs(config: &RunsConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_runs")?;

    let runs = if config.demo {
        generate_demo_runs()
    } else {
        println!("No training runs found. Use --demo for sample data.");
        return Ok(());
    };

    ctx.record_metric("run_count", runs.len() as i64);

    match config.subcommand.as_str() {
        "show" => {
            let run_id = config
                .run_id
                .as_deref()
                .unwrap_or_else(|| runs.first().map_or("run-001", |r| r.id.as_str()));
            print_run_detail(&runs, run_id)?;
        }
        "compare" => {
            let id_a = config.run_id.as_deref().unwrap_or("run-001");
            let id_b = config.compare_id.as_deref().unwrap_or("run-003");
            print_run_comparison(&runs, id_a, id_b)?;
        }
        _ => print_run_list(&runs),
    }

    Ok(())
}

fn generate_demo_runs() -> Vec<TrainingRun> {
    #[allow(clippy::type_complexity)]
    let specs: &[(&str, &str, &[(&str, &str)])] = &[
        (
            "run-001",
            "baseline-sgd",
            &[
                ("optimizer", "sgd"),
                ("learning_rate", "0.01"),
                ("batch_size", "32"),
                ("weight_decay", "0.0001"),
            ],
        ),
        (
            "run-002",
            "adam-default",
            &[
                ("optimizer", "adam"),
                ("learning_rate", "0.001"),
                ("batch_size", "64"),
                ("weight_decay", "0.0"),
            ],
        ),
        (
            "run-003",
            "adam-warmup",
            &[
                ("optimizer", "adam"),
                ("learning_rate", "0.0005"),
                ("batch_size", "64"),
                ("weight_decay", "0.01"),
                ("warmup_steps", "500"),
            ],
        ),
        (
            "run-004",
            "adamw-cosine",
            &[
                ("optimizer", "adamw"),
                ("learning_rate", "0.0003"),
                ("batch_size", "128"),
                ("weight_decay", "0.05"),
                ("scheduler", "cosine"),
            ],
        ),
    ];

    specs
        .iter()
        .map(|(id, name, params)| {
            let seed = run_seed(id);
            let hyperparams: HashMap<String, String> = params
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            TrainingRun {
                id: (*id).to_string(),
                name: (*name).to_string(),
                epoch: seed_to_epoch(seed),
                loss: seed_to_loss(seed),
                accuracy: seed_to_accuracy(seed),
                duration_secs: seed_to_duration(seed),
                timestamp: seed_to_timestamp(seed),
                hyperparams,
            }
        })
        .collect()
}

fn run_seed(id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

fn seed_to_epoch(seed: u64) -> u32 {
    5 + (seed % 46) as u32
}

fn seed_to_loss(seed: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    1u64.hash(&mut hasher);
    let h = hasher.finish();
    0.01 + (h % 200) as f64 / 1000.0
}

fn seed_to_accuracy(seed: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    2u64.hash(&mut hasher);
    let h = hasher.finish();
    0.80 + (h % 190) as f64 / 1000.0
}

fn seed_to_duration(seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    3u64.hash(&mut hasher);
    let h = hasher.finish();
    60 + (h % 3540)
}

fn seed_to_timestamp(seed: u64) -> String {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    4u64.hash(&mut hasher);
    let h = hasher.finish();
    let month = 1 + (h % 3);
    let day = 1 + ((h >> 8) % 28);
    let hour = (h >> 16) % 24;
    let minute = (h >> 24) % 60;
    format!("2026-{month:02}-{day:02} {hour:02}:{minute:02}")
}

fn format_duration(secs: u64) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn print_run_list(runs: &[TrainingRun]) {
    let mut sorted: Vec<&TrainingRun> = runs.iter().collect();
    sorted.sort_by(|a, b| {
        a.loss
            .partial_cmp(&b.loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("Training Runs (sorted by loss)");
    println!("==============================");
    println!();
    println!(
        "{:<10} {:<18} {:>6} {:>8} {:>10} {:>12} {:<18}",
        "ID", "NAME", "EPOCH", "LOSS", "ACCURACY", "DURATION", "TIMESTAMP"
    );
    println!("{}", "-".repeat(86));

    for run in &sorted {
        println!(
            "{:<10} {:<18} {:>6} {:>8.4} {:>9.1}% {:>12} {:<18}",
            run.id,
            run.name,
            run.epoch,
            run.loss,
            run.accuracy * 100.0,
            format_duration(run.duration_secs),
            run.timestamp
        );
    }

    // Best run indicator
    if let Some(best) = sorted.first() {
        println!();
        println!(
            "Best run: {} (loss={:.4}, accuracy={:.1}%)",
            best.id,
            best.loss,
            best.accuracy * 100.0
        );
    }
}

fn find_run<'a>(runs: &'a [TrainingRun], id: &str) -> Option<&'a TrainingRun> {
    runs.iter().find(|r| r.id == id)
}

fn print_run_detail(runs: &[TrainingRun], run_id: &str) -> Result<()> {
    let Some(run) = find_run(runs, run_id) else {
        println!("Run '{run_id}' not found.");
        println!("Available runs:");
        for r in runs {
            println!("  {}: {}", r.id, r.name);
        }
        return Ok(());
    };

    println!("Training Run Detail");
    println!("====================");
    println!();
    println!("ID:         {}", run.id);
    println!("Name:       {}", run.name);
    println!("Epoch:      {}", run.epoch);
    println!("Loss:       {:.4}", run.loss);
    println!("Accuracy:   {:.1}%", run.accuracy * 100.0);
    println!("Duration:   {}", format_duration(run.duration_secs));
    println!("Timestamp:  {}", run.timestamp);
    println!();
    println!("Hyperparameters:");
    println!("----------------");

    let mut keys: Vec<&String> = run.hyperparams.keys().collect();
    keys.sort();
    for key in &keys {
        if let Some(val) = run.hyperparams.get(*key) {
            println!("  {:<20} {}", key, val);
        }
    }

    Ok(())
}

fn print_run_comparison(runs: &[TrainingRun], id_a: &str, id_b: &str) -> Result<()> {
    let Some(run_a) = find_run(runs, id_a) else {
        println!("Run '{id_a}' not found.");
        return Ok(());
    };
    let Some(run_b) = find_run(runs, id_b) else {
        println!("Run '{id_b}' not found.");
        return Ok(());
    };

    println!("Training Run Comparison");
    println!("========================");
    println!();
    println!("{:<20} {:>18} {:>18} {:>10}", "METRIC", id_a, id_b, "DELTA");
    println!("{}", "-".repeat(68));

    // Name
    let name_marker = if run_a.name == run_b.name {
        " ".to_string()
    } else {
        "*".to_string()
    };
    println!(
        "{:<20} {:>18} {:>18} {:>10}",
        "name", run_a.name, run_b.name, name_marker
    );

    // Epoch
    let epoch_delta = i64::from(run_b.epoch) - i64::from(run_a.epoch);
    println!(
        "{:<20} {:>18} {:>18} {:>+10}",
        "epoch", run_a.epoch, run_b.epoch, epoch_delta
    );

    // Loss
    let loss_delta = run_b.loss - run_a.loss;
    let loss_indicator = if loss_delta < 0.0 { "better" } else { "worse" };
    println!(
        "{:<20} {:>18.4} {:>18.4} {:>+10.4}",
        "loss", run_a.loss, run_b.loss, loss_delta
    );

    // Accuracy
    let acc_delta = (run_b.accuracy - run_a.accuracy) * 100.0;
    let acc_indicator = if acc_delta > 0.0 { "better" } else { "worse" };
    println!(
        "{:<20} {:>17.1}% {:>17.1}% {:>+9.1}%",
        "accuracy",
        run_a.accuracy * 100.0,
        run_b.accuracy * 100.0,
        acc_delta
    );

    // Duration
    let dur_delta = run_b.duration_secs as i64 - run_a.duration_secs as i64;
    println!(
        "{:<20} {:>18} {:>18} {:>+10}s",
        "duration",
        format_duration(run_a.duration_secs),
        format_duration(run_b.duration_secs),
        dur_delta
    );

    println!();
    println!("Hyperparameter Differences:");
    println!("--------------------------");

    // Collect all keys from both runs
    let mut all_keys: Vec<String> = run_a
        .hyperparams
        .keys()
        .chain(run_b.hyperparams.keys())
        .cloned()
        .collect();
    all_keys.sort();
    all_keys.dedup();

    for key in &all_keys {
        let val_a = run_a.hyperparams.get(key).map_or("-", String::as_str);
        let val_b = run_b.hyperparams.get(key).map_or("-", String::as_str);
        let changed = if val_a == val_b { " " } else { "*" };
        println!("  {changed} {:<20} {:>14} {:>14}", key, val_a, val_b);
    }

    println!();
    println!(
        "Verdict: {} has {} loss and {} accuracy.",
        id_b, loss_indicator, acc_indicator
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_defaults() {
        let args = vec!["apr-runs".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.subcommand, "list");
        assert!(!config.demo);
        assert!(config.run_id.is_none());
    }

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-runs".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert!(config.demo);
    }

    #[test]
    fn test_parse_args_show() {
        let args = vec![
            "apr-runs".to_string(),
            "show".to_string(),
            "run-001".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.subcommand, "show");
        assert_eq!(config.run_id, Some("run-001".to_string()));
    }

    #[test]
    fn test_parse_args_compare() {
        let args = vec![
            "apr-runs".to_string(),
            "compare".to_string(),
            "run-001".to_string(),
            "run-003".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.subcommand, "compare");
        assert_eq!(config.run_id, Some("run-001".to_string()));
        assert_eq!(config.compare_id, Some("run-003".to_string()));
    }

    #[test]
    fn test_parse_args_unknown() {
        let args = vec!["apr-runs".to_string(), "--badarg".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_generate_demo_runs_count() {
        let runs = generate_demo_runs();
        assert_eq!(runs.len(), 4);
    }

    #[test]
    fn test_generate_demo_runs_deterministic() {
        let r1 = generate_demo_runs();
        let r2 = generate_demo_runs();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.loss, b.loss);
            assert_eq!(a.accuracy, b.accuracy);
            assert_eq!(a.epoch, b.epoch);
        }
    }

    #[test]
    fn test_generate_demo_runs_valid_ranges() {
        let runs = generate_demo_runs();
        for run in &runs {
            assert!(run.epoch >= 5);
            assert!(run.loss >= 0.01 && run.loss < 0.22);
            assert!(run.accuracy >= 0.80 && run.accuracy < 1.0);
            assert!(run.duration_secs >= 60);
            assert!(!run.hyperparams.is_empty());
        }
    }

    #[test]
    fn test_find_run_exists() {
        let runs = generate_demo_runs();
        let found = find_run(&runs, "run-001");
        assert!(found.is_some());
        assert_eq!(found.map(|r| r.id.as_str()), Some("run-001"));
    }

    #[test]
    fn test_find_run_missing() {
        let runs = generate_demo_runs();
        assert!(find_run(&runs, "run-999").is_none());
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(45), "45s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(125), "2m 5s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3661), "1h 1m 1s");
    }

    #[test]
    fn test_run_list_demo() {
        let config = RunsConfig {
            subcommand: "list".to_string(),
            run_id: None,
            compare_id: None,
            demo: true,
        };
        assert!(run_runs(&config).is_ok());
    }

    #[test]
    fn test_run_show_demo() {
        let config = RunsConfig {
            subcommand: "show".to_string(),
            run_id: Some("run-001".to_string()),
            compare_id: None,
            demo: true,
        };
        assert!(run_runs(&config).is_ok());
    }

    #[test]
    fn test_run_compare_demo() {
        let config = RunsConfig {
            subcommand: "compare".to_string(),
            run_id: Some("run-001".to_string()),
            compare_id: Some("run-003".to_string()),
            demo: true,
        };
        assert!(run_runs(&config).is_ok());
    }

    #[test]
    fn test_run_show_missing_id() {
        let config = RunsConfig {
            subcommand: "show".to_string(),
            run_id: Some("run-999".to_string()),
            compare_id: None,
            demo: true,
        };
        assert!(run_runs(&config).is_ok());
    }

    #[test]
    fn test_hyperparams_vary_across_runs() {
        let runs = generate_demo_runs();
        let optimizers: Vec<&str> = runs
            .iter()
            .filter_map(|r| r.hyperparams.get("optimizer").map(|s| s.as_str()))
            .collect();
        // At least two different optimizers
        let unique: std::collections::HashSet<&&str> = optimizers.iter().collect();
        assert!(unique.len() >= 2);
    }

    #[test]
    fn test_seed_to_timestamp_deterministic() {
        let t1 = seed_to_timestamp(42);
        let t2 = seed_to_timestamp(42);
        assert_eq!(t1, t2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_seed_to_loss_in_range(seed in 0u64..1_000_000) {
            let loss = seed_to_loss(seed);
            prop_assert!(loss >= 0.01);
            prop_assert!(loss < 0.22);
        }

        #[test]
        fn prop_seed_to_accuracy_in_range(seed in 0u64..1_000_000) {
            let acc = seed_to_accuracy(seed);
            prop_assert!(acc >= 0.80);
            prop_assert!(acc < 1.0);
        }

        #[test]
        fn prop_format_duration_non_empty(secs in 0u64..100_000) {
            let formatted = format_duration(secs);
            prop_assert!(!formatted.is_empty());
        }

        #[test]
        fn prop_seed_to_epoch_in_range(seed in 0u64..1_000_000) {
            let epoch = seed_to_epoch(seed);
            prop_assert!(epoch >= 5);
            prop_assert!(epoch <= 50);
        }
    }
}
