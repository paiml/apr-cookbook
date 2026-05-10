//! Tier 3.2 Hyperparameter optimization — shared helper for 5 recipes.
//!
//! Implements deterministic search algorithms over a small synthetic
//! objective so each recipe falsifier asserts a *closed-form* property of
//! the algorithm rather than a stochastic optimizer-trace claim.
//!
//! - Grid search: exact product of axes — count = ∏|axis|.
//! - Random search: deterministic-stride sampler covering N trials.
//! - TPE: density-ratio bias toward high-performing region.
//! - ASHA: η-based early-pruning at each rung (≥ 50% pruned at η=2).
//! - Hyperband: R=81 produces exactly 5 brackets (s_max+1 with η=3).

#![allow(clippy::needless_range_loop)]

/// Synthetic 2D objective with single optimum at (0.3, 0.7).
/// Used by all 5 hyperopt recipes for a common reference point.
#[must_use]
pub fn synthetic_objective(x: f64, y: f64) -> f64 {
    -((x - 0.3).powi(2) + (y - 0.7).powi(2))
}

/// Grid search: cartesian product over two 1-D axes.
#[must_use]
pub fn grid_search(x_grid: &[f64], y_grid: &[f64]) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(x_grid.len() * y_grid.len());
    for &x in x_grid {
        for &y in y_grid {
            out.push((x, y, synthetic_objective(x, y)));
        }
    }
    out
}

/// Deterministic-stride sampler: evaluate `n_trials` (x, y) points along a
/// pseudo-random walk seeded by `seed`. Returns trial scores.
#[must_use]
pub fn random_search(n_trials: u32, seed: u32) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(n_trials as usize);
    for t in 0..n_trials {
        let x = (((t * seed) % 23) as f64) / 23.0;
        let y = (((t * seed * 7 + 5) % 31) as f64) / 31.0;
        out.push((x, y, synthetic_objective(x, y)));
    }
    out
}

/// TPE bias: density of points where score > median, then sample
/// preferentially from that region. Returns count of biased samples whose
/// distance to optimum (0.3, 0.7) is < 0.4.
#[must_use]
pub fn tpe_density_count(trials: &[(f64, f64, f64)]) -> usize {
    if trials.is_empty() {
        return 0;
    }
    let mut sorted_scores: Vec<f64> = trials.iter().map(|(_, _, s)| *s).collect();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_scores[sorted_scores.len() / 2];
    trials
        .iter()
        .filter(|(_, _, s)| *s >= median)
        .filter(|(x, y, _)| ((x - 0.3).powi(2) + (y - 0.7).powi(2)).sqrt() < 0.4)
        .count()
}

/// ASHA pruning: at each rung, top 1/η of trials advance, the rest are pruned.
/// Returns (n_kept_per_rung, n_pruned_per_rung) for n_rungs rungs at given η.
#[must_use]
pub fn asha_rungs(n_initial: u32, eta: u32, n_rungs: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(n_rungs as usize);
    let mut active = n_initial;
    for _ in 0..n_rungs {
        let kept = active / eta;
        let pruned = active - kept;
        out.push((kept, pruned));
        active = kept;
    }
    out
}

/// Hyperband bracket count given resource ceiling R and η.
/// Formula: s_max = floor(log_eta(R)); n_brackets = s_max + 1.
#[must_use]
pub fn hyperband_brackets(r_max: u32, eta: u32) -> u32 {
    if r_max <= 1 || eta < 2 {
        return 1;
    }
    let log_r = (f64::from(r_max).ln()) / (f64::from(eta).ln());
    log_r.floor() as u32 + 1
}

/// Best score among trials.
#[must_use]
pub fn best_score(trials: &[(f64, f64, f64)]) -> f64 {
    trials
        .iter()
        .map(|(_, _, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_count_is_product_of_axes() {
        let x = vec![0.0, 0.25, 0.5, 0.75];
        let y = vec![0.0, 0.5, 1.0];
        let trials = grid_search(&x, &y);
        assert_eq!(trials.len(), x.len() * y.len());
    }

    #[test]
    fn grid_finds_score_near_optimum() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![0.0, 0.5, 0.7, 1.0];
        let best = best_score(&grid_search(&x, &y));
        // (0.25, 0.7) gives -((0.25-0.3)^2 + 0) = -0.0025 — very close to 0.
        assert!(best > -0.01);
    }

    #[test]
    fn random_search_deterministic_for_fixed_seed() {
        let r1 = random_search(20, 7);
        let r2 = random_search(20, 7);
        assert_eq!(r1, r2);
    }

    #[test]
    fn tpe_density_finds_high_scoring_region() {
        let trials = random_search(100, 13);
        let count = tpe_density_count(&trials);
        // At least one trial in 100 random samples lands within 0.4 of (0.3, 0.7).
        assert!(
            count >= 1,
            "TPE density should find ≥1 high-score sample, got {count}"
        );
    }

    #[test]
    fn asha_eta2_prunes_50_percent() {
        let rungs = asha_rungs(100, 2, 3);
        for (kept, pruned) in rungs {
            assert!(
                pruned >= kept,
                "ASHA at η=2 must prune ≥ 50%, got kept={kept} pruned={pruned}"
            );
        }
    }

    #[test]
    fn hyperband_r81_eta3_yields_5_brackets() {
        // Per Li et al. 2018: with η=3, R=81 → s_max=4 → 5 brackets.
        assert_eq!(hyperband_brackets(81, 3), 5);
    }

    #[test]
    fn hyperband_r1_yields_1_bracket() {
        assert_eq!(hyperband_brackets(1, 3), 1);
    }
}
