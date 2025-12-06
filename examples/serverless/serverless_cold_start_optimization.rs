//! # Recipe: Cold Start Optimization
//!
//! **Category**: Serverless/Lambda
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
//! Optimize cold start latency for serverless model deployment.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serverless_cold_start_optimization
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("serverless_cold_start_optimization")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Cold start optimization strategies");
    println!();

    // Baseline: No optimization
    let baseline = measure_cold_start(ColdStartConfig {
        model_size_mb: 50,
        lazy_loading: false,
        model_caching: false,
        warmup_enabled: false,
        provisioned_concurrency: 0,
    });

    println!("Baseline (no optimization):");
    println!("  Init time: {}ms", baseline.init_time_ms);
    println!("  First request: {}ms", baseline.first_request_ms);
    println!("  Total cold start: {}ms", baseline.total_cold_start_ms);
    println!();

    // Strategy 1: Lazy loading
    let lazy = measure_cold_start(ColdStartConfig {
        model_size_mb: 50,
        lazy_loading: true,
        model_caching: false,
        warmup_enabled: false,
        provisioned_concurrency: 0,
    });

    println!("Strategy 1 - Lazy Loading:");
    println!(
        "  Init time: {}ms (↓{}ms)",
        lazy.init_time_ms,
        baseline.init_time_ms - lazy.init_time_ms
    );
    println!("  First request: {}ms", lazy.first_request_ms);
    println!();

    // Strategy 2: Model caching
    let cached = measure_cold_start(ColdStartConfig {
        model_size_mb: 50,
        lazy_loading: true,
        model_caching: true,
        warmup_enabled: false,
        provisioned_concurrency: 0,
    });

    println!("Strategy 2 - Model Caching:");
    println!("  Init time: {}ms", cached.init_time_ms);
    println!(
        "  First request: {}ms (↓{}ms)",
        cached.first_request_ms,
        lazy.first_request_ms - cached.first_request_ms
    );
    println!();

    // Strategy 3: Warmup
    let warmed = measure_cold_start(ColdStartConfig {
        model_size_mb: 50,
        lazy_loading: true,
        model_caching: true,
        warmup_enabled: true,
        provisioned_concurrency: 0,
    });

    println!("Strategy 3 - Warmup Enabled:");
    println!("  Init time: {}ms", warmed.init_time_ms);
    println!(
        "  First request: {}ms (↓{}ms)",
        warmed.first_request_ms,
        cached.first_request_ms - warmed.first_request_ms
    );
    println!();

    // Strategy 4: Provisioned concurrency
    let provisioned = measure_cold_start(ColdStartConfig {
        model_size_mb: 50,
        lazy_loading: true,
        model_caching: true,
        warmup_enabled: true,
        provisioned_concurrency: 5,
    });

    println!("Strategy 4 - Provisioned Concurrency:");
    println!(
        "  Cold starts eliminated: {}",
        provisioned.cold_starts_eliminated
    );
    println!(
        "  Effective cold start: {}ms",
        provisioned.total_cold_start_ms
    );
    println!();

    // Summary
    let improvement = (f64::from(baseline.total_cold_start_ms - provisioned.total_cold_start_ms)
        / f64::from(baseline.total_cold_start_ms))
        * 100.0;

    ctx.record_metric("baseline_ms", i64::from(baseline.total_cold_start_ms));
    ctx.record_metric("optimized_ms", i64::from(provisioned.total_cold_start_ms));
    ctx.record_float_metric("improvement_pct", improvement);

    println!("Summary:");
    println!("  Baseline: {}ms", baseline.total_cold_start_ms);
    println!("  Optimized: {}ms", provisioned.total_cold_start_ms);
    println!("  Improvement: {:.1}%", improvement);

    // Save optimization report
    let report_path = ctx.path("cold_start_report.json");
    save_report(&report_path, &[baseline, lazy, cached, warmed, provisioned])?;
    println!();
    println!("Report saved to: {:?}", report_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdStartConfig {
    model_size_mb: u32,
    lazy_loading: bool,
    model_caching: bool,
    warmup_enabled: bool,
    provisioned_concurrency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdStartMetrics {
    config: ColdStartConfig,
    init_time_ms: u32,
    first_request_ms: u32,
    total_cold_start_ms: u32,
    cold_starts_eliminated: bool,
}

fn measure_cold_start(config: ColdStartConfig) -> ColdStartMetrics {
    // Deterministic simulation of cold start times
    let base_init = config.model_size_mb * 2; // 2ms per MB

    let init_time = if config.lazy_loading {
        base_init / 4 // Lazy loading reduces init by 75%
    } else {
        base_init
    };

    let first_request = if config.model_caching {
        20 // Cached model loads fast
    } else if config.lazy_loading {
        base_init // Load on first request
    } else {
        30 // Already loaded
    };

    let warmup_reduction = if config.warmup_enabled { 10 } else { 0 };

    let cold_starts_eliminated = config.provisioned_concurrency > 0;
    let total = if cold_starts_eliminated {
        0 // Provisioned concurrency eliminates cold starts
    } else {
        init_time + first_request - warmup_reduction
    };

    ColdStartMetrics {
        config,
        init_time_ms: init_time,
        first_request_ms: first_request - warmup_reduction,
        total_cold_start_ms: total,
        cold_starts_eliminated,
    }
}

fn save_report(path: &std::path::Path, metrics: &[ColdStartMetrics]) -> Result<()> {
    let json = serde_json::to_string_pretty(metrics)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_cold_start() {
        let metrics = measure_cold_start(ColdStartConfig {
            model_size_mb: 50,
            lazy_loading: false,
            model_caching: false,
            warmup_enabled: false,
            provisioned_concurrency: 0,
        });

        assert!(metrics.total_cold_start_ms > 0);
        assert!(!metrics.cold_starts_eliminated);
    }

    #[test]
    fn test_lazy_loading_reduces_init() {
        let baseline = measure_cold_start(ColdStartConfig {
            model_size_mb: 100,
            lazy_loading: false,
            model_caching: false,
            warmup_enabled: false,
            provisioned_concurrency: 0,
        });

        let lazy = measure_cold_start(ColdStartConfig {
            model_size_mb: 100,
            lazy_loading: true,
            model_caching: false,
            warmup_enabled: false,
            provisioned_concurrency: 0,
        });

        assert!(lazy.init_time_ms < baseline.init_time_ms);
    }

    #[test]
    fn test_provisioned_eliminates_cold_start() {
        let metrics = measure_cold_start(ColdStartConfig {
            model_size_mb: 50,
            lazy_loading: true,
            model_caching: true,
            warmup_enabled: true,
            provisioned_concurrency: 5,
        });

        assert!(metrics.cold_starts_eliminated);
        assert_eq!(metrics.total_cold_start_ms, 0);
    }

    #[test]
    fn test_deterministic_metrics() {
        let config = ColdStartConfig {
            model_size_mb: 50,
            lazy_loading: true,
            model_caching: false,
            warmup_enabled: false,
            provisioned_concurrency: 0,
        };

        let m1 = measure_cold_start(config.clone());
        let m2 = measure_cold_start(config);

        assert_eq!(m1.total_cold_start_ms, m2.total_cold_start_ms);
    }

    #[test]
    fn test_save_report() {
        let ctx = RecipeContext::new("test_cold_start_report").unwrap();
        let path = ctx.path("report.json");

        let metrics = vec![measure_cold_start(ColdStartConfig {
            model_size_mb: 10,
            lazy_loading: false,
            model_caching: false,
            warmup_enabled: false,
            provisioned_concurrency: 0,
        })];

        save_report(&path, &metrics).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_lazy_always_reduces_init(model_size in 10u32..200) {
            let baseline = measure_cold_start(ColdStartConfig {
                model_size_mb: model_size,
                lazy_loading: false,
                model_caching: false,
                warmup_enabled: false,
                provisioned_concurrency: 0,
            });

            let lazy = measure_cold_start(ColdStartConfig {
                model_size_mb: model_size,
                lazy_loading: true,
                model_caching: false,
                warmup_enabled: false,
                provisioned_concurrency: 0,
            });

            prop_assert!(lazy.init_time_ms <= baseline.init_time_ms);
        }

        #[test]
        fn prop_provisioned_always_zero(model_size in 10u32..200, concurrency in 1u32..10) {
            let metrics = measure_cold_start(ColdStartConfig {
                model_size_mb: model_size,
                lazy_loading: true,
                model_caching: true,
                warmup_enabled: true,
                provisioned_concurrency: concurrency,
            });

            prop_assert_eq!(metrics.total_cold_start_ms, 0);
            prop_assert!(metrics.cold_starts_eliminated);
        }
    }
}
