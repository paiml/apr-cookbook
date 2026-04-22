//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]

#[cfg(test)]
mod tests {
    use super::super::*;

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
    use super::super::*;
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
