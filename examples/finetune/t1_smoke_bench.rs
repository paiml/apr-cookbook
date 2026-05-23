//! # Tier 1.5 — Smoke — Bench
//!
//! Falsifier: apr bench finetune produces a deterministic per-step
//! latency histogram.
//!
//! Run with: cargo run --example t1_smoke_bench

use apr_cookbook::finetune::smoke;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_smoke_bench")?;
    // Synthetic deterministic latencies.
    let latencies: Vec<u64> = (1..=100u64).map(|i| i * 100).collect();
    let h = smoke::compute_bench_histogram(&latencies);
    println!(
        "✓ bench histogram: steps={}, p50={}us, p95={}us, p99={}us",
        h.step_count, h.p50_us, h.p95_us, h.p99_us
    );
    assert!(
        h.p50_us <= h.p95_us && h.p95_us <= h.p99_us,
        "histogram should be monotone-non-decreasing percentiles"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let l: Vec<u64> = (1..=100u64).map(|i| i * 100).collect();
        let a = smoke::compute_bench_histogram(&l);
        let b = smoke::compute_bench_histogram(&l);
        assert_eq!(a, b);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty input gives a zero histogram — distinct from non-empty.
        let h0 = smoke::compute_bench_histogram(&[]);
        let h1 = smoke::compute_bench_histogram(&[100, 200, 300]);
        assert_ne!(h0, h1);
    }

    #[test]
    fn deterministic_across_runs() {
        let l: Vec<u64> = (1..=10u64).collect();
        assert_eq!(
            smoke::compute_bench_histogram(&l),
            smoke::compute_bench_histogram(&l)
        );
    }
}
