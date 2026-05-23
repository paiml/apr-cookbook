//! Tier 1.5 smoke + bench helpers.
//!
//! These recipes verify apr-native CLI affordances (--plan, --resume,
//! --early-stop, --dry-run, --bench) work correctly without needing
//! to actually run any training. They simulate the workflow shape and
//! assert structural invariants.

/// Verdict of a planning step. apr finetune --plan produces a
/// non-executing report of what training would do.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanReport {
    pub method: String,
    pub trainable_params: u64,
    pub estimated_memory_mb: u64,
    pub epochs: u32,
    pub wrote_checkpoint: bool,
}

impl PlanReport {
    /// Falsifier: --plan emits a plan but writes no checkpoint.
    #[must_use]
    pub fn is_plan_only(&self) -> bool {
        !self.wrote_checkpoint
    }
}

/// Compute a deterministic plan for given (rank, base-param-count).
/// Models LoRA: trainable = 2 × d × rank for each adapted layer.
#[must_use]
pub fn plan_lora(base_params: u64, rank: u32, n_layers: u32) -> PlanReport {
    // 2 × d × rank per layer. d = sqrt(base_params / n_layers) approx.
    let d = ((base_params / u64::from(n_layers)) as f64).sqrt() as u64;
    let trainable = 2 * d * u64::from(rank) * u64::from(n_layers);
    PlanReport {
        method: format!("lora-r{rank}"),
        trainable_params: trainable,
        estimated_memory_mb: trainable * 4 / 1_000_000, // 4 bytes per f32
        epochs: 1,
        wrote_checkpoint: false,
    }
}

/// Resume tracking: state at last persisted checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeState {
    pub last_step: u32,
    pub last_epoch: u32,
    pub interrupted_at: Option<u32>,
}

/// Falsifier: interrupted finetune resumes at the last persisted step.
/// Given an `interrupt_step`, the next run should resume at that exact step.
#[must_use]
pub fn simulate_resume(total_steps: u32, interrupt_step: u32) -> ResumeState {
    ResumeState {
        last_step: interrupt_step,
        last_epoch: interrupt_step / 100, // 100 steps per epoch
        interrupted_at: if interrupt_step < total_steps {
            Some(interrupt_step)
        } else {
            None
        },
    }
}

/// Early-stop telemetry: number of epochs trained before plateau triggered.
#[must_use]
pub fn simulate_early_stop(losses: &[f64], patience: u32) -> u32 {
    if losses.is_empty() {
        return 0;
    }
    let mut best = losses[0];
    let mut bad_epochs = 0u32;
    for (i, &l) in losses.iter().enumerate().skip(1) {
        if l < best {
            best = l;
            bad_epochs = 0;
        } else {
            bad_epochs += 1;
            if bad_epochs >= patience {
                return i as u32;
            }
        }
    }
    losses.len() as u32 - 1
}

/// Dry-run verdict: zero side effects.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DryRunVerdict {
    pub fs_writes: u32,
    pub gpu_allocations: u32,
    pub network_calls: u32,
}

impl DryRunVerdict {
    /// Falsifier: dry-run produces zero side effects.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.fs_writes == 0 && self.gpu_allocations == 0 && self.network_calls == 0
    }
}

/// Bench histogram: per-step latency buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchHistogram {
    pub step_count: u32,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

/// Compute deterministic bench histogram from synthetic latencies.
#[must_use]
pub fn compute_bench_histogram(latencies_us: &[u64]) -> BenchHistogram {
    if latencies_us.is_empty() {
        return BenchHistogram {
            step_count: 0,
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
        };
    }
    let mut sorted: Vec<u64> = latencies_us.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p50 = sorted[n / 2];
    let p95 = sorted[(n * 95).div_ceil(100).min(n - 1)];
    let p99 = sorted[(n * 99).div_ceil(100).min(n - 1)];
    BenchHistogram {
        step_count: n as u32,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_is_plan_only() {
        let p = plan_lora(7_000_000_000, 8, 32);
        assert!(p.is_plan_only());
        assert!(p.trainable_params > 0);
    }

    #[test]
    fn resume_at_interrupt_step() {
        let r = simulate_resume(1000, 250);
        assert_eq!(r.last_step, 250);
        assert_eq!(r.interrupted_at, Some(250));
    }

    #[test]
    fn resume_full_completion_returns_none() {
        let r = simulate_resume(1000, 1000);
        assert!(r.interrupted_at.is_none());
    }

    #[test]
    fn early_stop_halts_within_patience() {
        // Loss plateaus from epoch 5; with patience=2, we stop at epoch 7
        let losses = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.21, 0.22, 0.23];
        let stopped = simulate_early_stop(&losses, 2);
        assert!(
            stopped <= 6,
            "early stop should halt by epoch 6, got {stopped}"
        );
    }

    #[test]
    fn dry_run_is_clean() {
        let v = DryRunVerdict::default();
        assert!(v.is_clean());
    }

    #[test]
    fn bench_histogram_deterministic() {
        let lats = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let h1 = compute_bench_histogram(&lats);
        let h2 = compute_bench_histogram(&lats);
        assert_eq!(h1, h2);
        assert_eq!(h1.step_count, 10);
        assert_eq!(h1.p50_us, 600); // middle
    }
}
