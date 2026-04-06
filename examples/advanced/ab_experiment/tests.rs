#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

#[test]
fn test_full_experiment_succeeds() {
    let mut ctx = RecipeContext::new("test_ab_full").expect("context");
    let report = run_experiment(&mut ctx).expect("experiment");
    assert_eq!(report.results_a.len(), 200);
    assert_eq!(report.results_b.len(), 200);
}

#[test]
fn test_experiment_deterministic() {
    let mut ctx1 = RecipeContext::new("test_ab_determinism").expect("ctx1");
    let mut ctx2 = RecipeContext::new("test_ab_determinism").expect("ctx2");
    let r1 = run_experiment(&mut ctx1).expect("r1");
    let r2 = run_experiment(&mut ctx2).expect("r2");
    assert_eq!(r1.diff.accuracy_a, r2.diff.accuracy_a);
    assert_eq!(r1.diff.accuracy_b, r2.diff.accuracy_b);
    assert_eq!(r1.t_statistic, r2.t_statistic);
    assert_eq!(r1.verdict, r2.verdict);
}

#[test]
fn test_verdict_is_promote() {
    let mut ctx = RecipeContext::new("test_ab_promote").expect("context");
    let report = run_experiment(&mut ctx).expect("experiment");
    // Model B is configured to be better (0.80 vs 0.72 accuracy)
    assert_eq!(
        report.verdict,
        ExperimentVerdict::Promote,
        "candidate should be promoted (higher accuracy)"
    );
}

#[test]
fn test_ground_truth_balanced() {
    let mut ctx = RecipeContext::new("test_ab_gt").expect("context");
    let gt = generate_ground_truth(ctx.rng(), 1000);
    let ones = gt.iter().filter(|&&v| v > 0.5).count();
    // Should be roughly balanced (within 10% of 500)
    assert!(ones > 400, "too few positives: {}", ones);
    assert!(ones < 600, "too many positives: {}", ones);
}

#[test]
fn test_sample_results_valid_ranges() {
    let mut ctx = RecipeContext::new("test_ab_ranges").expect("context");
    let gt = generate_ground_truth(ctx.rng(), 50);
    let results = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
    for r in &results {
        assert!(
            r.confidence >= 0.0 && r.confidence <= 1.0,
            "confidence out of range: {}",
            r.confidence
        );
        assert!(
            r.latency_ms > 0.0,
            "latency must be positive: {}",
            r.latency_ms
        );
        assert!(
            r.prediction == 0.0 || r.prediction == 1.0,
            "prediction must be binary: {}",
            r.prediction
        );
    }
}

#[test]
fn test_diff_match_rate_range() {
    let mut ctx = RecipeContext::new("test_ab_diff_range").expect("context");
    let gt = generate_ground_truth(ctx.rng(), 100);
    let ra = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
    let rb = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
    let diff = compute_diff(&ra, &rb);
    assert!(
        diff.match_rate >= 0.0 && diff.match_rate <= 1.0,
        "match_rate out of range: {}",
        diff.match_rate
    );
}

#[test]
fn test_paired_t_statistic_identical() {
    // When two result sets are identical, confidence diffs are all zero
    let mut ctx = RecipeContext::new("test_ab_t_identical").expect("context");
    let gt = generate_ground_truth(ctx.rng(), 50);
    let results = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
    let t = paired_t_statistic(&results, &results);
    assert!(
        t.abs() < 1e-9,
        "t-statistic for identical sets should be ~0, got {}",
        t
    );
}

#[test]
fn test_determine_verdict_keep_when_a_better() {
    let diff = DiffResult {
        match_rate: 0.5,
        mean_confidence_delta: -0.1,
        mean_latency_delta: 0.0,
        accuracy_a: 0.90,
        accuracy_b: 0.70,
    };
    let config = setup_config();
    // Large negative t-stat => A is significantly better
    let verdict = determine_verdict(&diff, -5.0, &config);
    assert_eq!(verdict, ExperimentVerdict::Keep);
}

#[test]
fn test_determine_verdict_inconclusive() {
    let diff = DiffResult {
        match_rate: 0.9,
        mean_confidence_delta: 0.001,
        mean_latency_delta: 0.0,
        accuracy_a: 0.80,
        accuracy_b: 0.81,
    };
    let config = setup_config();
    // Small t-stat => not significant
    let verdict = determine_verdict(&diff, 0.5, &config);
    assert_eq!(verdict, ExperimentVerdict::Inconclusive);
}

#[test]
fn test_mean_latency_empty() {
    assert!((mean_latency(&[]) - 0.0).abs() < 1e-12);
}
