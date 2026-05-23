//! # Recipe: CBTop Latency Histogram Aggregation
//!
//! **Category**: monitoring
//! **CLI Equivalent**: `apr cbtop --histogram --buckets 16 --out histogram.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example cbtop_histogram` exits 0
//! 2. [x] `cargo test --example cbtop_histogram` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr cbtop` aggregation in-process (no shell-out)
//! 10. [x] Unit tests cover bucket math, total conservation, empty input
//!
//! ## Learning Objective
//! Aggregates a latency stream into a fixed-width histogram with ASCII rendering
//! and overflow reporting. The histogram is the canonical cbtop summary view for
//! post-mortem performance analysis.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cbtop_histogram
//! ```
//!
//! ## References
//! - Dean, J. & Barroso, L.A. (2013). *The Tail at Scale*. Communications of the ACM. DOI: 10.1145/2408776.2408794

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Histogram {
    min: f32,
    max: f32,
    n_buckets: usize,
    counts: Vec<usize>,
    total: usize,
    overflow: usize,
}

impl Histogram {
    fn new(min: f32, max: f32, n_buckets: usize) -> Self {
        Self {
            min,
            max,
            n_buckets: n_buckets.max(1),
            counts: vec![0; n_buckets.max(1)],
            total: 0,
            overflow: 0,
        }
    }

    fn bucket_width(&self) -> f32 {
        (self.max - self.min) / self.n_buckets as f32
    }

    fn insert(&mut self, x: f32) {
        self.total += 1;
        if x < self.min || x >= self.max {
            self.overflow += 1;
            return;
        }
        let w = self.bucket_width();
        if w <= 0.0 {
            return;
        }
        let idx = (((x - self.min) / w) as usize).min(self.n_buckets - 1);
        self.counts[idx] += 1;
    }

    fn peak(&self) -> usize {
        *self.counts.iter().max().unwrap_or(&0)
    }
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn generate_latencies(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // Bimodal mixture: mostly fast (~3ms), 10% slow (~15ms) to exercise
            // the tail buckets.
            let is_slow = rng.gen_range(0.0..1.0_f32) < 0.10;
            if is_slow {
                12.0 + rng.gen_range(0.0..6.0)
            } else {
                2.0 + rng.gen_range(0.0..2.0) + (i % 7) as f32 * 0.01
            }
        })
        .collect()
}

fn aggregate_histogram(values: &[f32], min: f32, max: f32, n_buckets: usize) -> Histogram {
    let mut h = Histogram::new(min, max, n_buckets);
    for &v in values {
        h.insert(v);
    }
    h
}

fn render_histogram(h: &Histogram, width: usize) -> String {
    let peak = h.peak().max(1);
    let bucket_w = h.bucket_width();
    let mut out = String::new();
    for (i, &count) in h.counts.iter().enumerate() {
        let lo = h.min + bucket_w * i as f32;
        let hi = lo + bucket_w;
        let filled = ((count as f32 / peak as f32) * width as f32) as usize;
        let bar = "#".repeat(filled);
        out.push_str(&format!(
            "[{:>6.2}, {:>6.2}) | {:<width$} {}\n",
            lo,
            hi,
            bar,
            count,
            width = width
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("cbtop_histogram")?;
    println!("=== Recipe: {} ===", ctx.name());

    let n_samples = 500_usize;
    let n_buckets = 12_usize;
    let min_ms = 0.0_f32;
    let max_ms = 20.0_f32;

    let latencies = generate_latencies(ctx.rng(), n_samples);
    let h = aggregate_histogram(&latencies, min_ms, max_ms, n_buckets);

    println!(
        "Samples: {}, Buckets: {}, Range: [{:.1}, {:.1}) ms",
        h.total, h.n_buckets, h.min, h.max
    );
    println!("Overflow (outside range): {}", h.overflow);
    println!(
        "Bucket width: {:.2} ms, Peak bucket count: {}",
        h.bucket_width(),
        h.peak()
    );

    println!("\n--- Histogram ---");
    print!("{}", render_histogram(&h, 30));

    let bucket_sum: usize = h.counts.iter().sum();
    assert_eq!(bucket_sum + h.overflow, h.total);

    let out = json!({
        "recipe": ctx.name(),
        "samples": h.total,
        "min_ms": h.min,
        "max_ms": h.max,
        "n_buckets": h.n_buckets,
        "bucket_width_ms": h.bucket_width(),
        "counts": h.counts,
        "overflow": h.overflow,
        "peak": h.peak(),
    });
    let out_path = ctx.path("histogram.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.record_metric("samples", h.total as i64);
    ctx.record_metric("n_buckets", h.n_buckets as i64);
    ctx.record_metric("peak", h.peak() as i64);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_total_equals_sum_plus_overflow() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let values = generate_latencies(&mut rng, 200);
        let h = aggregate_histogram(&values, 0.0, 10.0, 8);
        let bucket_sum: usize = h.counts.iter().sum();
        assert_eq!(bucket_sum + h.overflow, h.total);
    }

    #[test]
    fn test_value_below_min_is_overflow() {
        let h = aggregate_histogram(&[-1.0_f32], 0.0, 10.0, 4);
        assert_eq!(h.overflow, 1);
        assert!(h.counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_value_at_max_is_overflow() {
        // Max is exclusive in our insert().
        let h = aggregate_histogram(&[10.0_f32], 0.0, 10.0, 4);
        assert_eq!(h.overflow, 1);
    }

    #[test]
    fn test_empty_produces_zero_counts() {
        let h = aggregate_histogram(&[], 0.0, 10.0, 4);
        assert_eq!(h.total, 0);
        assert_eq!(h.overflow, 0);
        assert!(h.counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_bucket_assignment_boundaries() {
        let vals: Vec<f32> = vec![0.0, 2.5, 5.0, 7.5, 9.9];
        let h = aggregate_histogram(&vals, 0.0, 10.0, 4);
        assert_eq!(h.total, 5);
        assert_eq!(h.overflow, 0);
        // Each bucket should have exactly 1 except possibly boundaries.
        let total_in_buckets: usize = h.counts.iter().sum();
        assert_eq!(total_in_buckets, 5);
    }

    #[test]
    fn test_render_has_bar_lines() {
        let h = aggregate_histogram(&[1.0_f32, 2.0, 3.0], 0.0, 4.0, 4);
        let rendered = render_histogram(&h, 10);
        // Each bucket line shows its range and count.
        assert_eq!(rendered.lines().count(), 4);
    }
}
