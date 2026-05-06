//! # apr rosetta diff-tensors — `--show-values <N>` Sampler
//!
//! `apr rosetta diff-tensors --show-values <N>` augments the layout report
//! with up to N representative element-pair samples per tensor. The
//! sampler must be deterministic (same input → same samples) so CI logs
//! are diff-able, and must include the extreme positions (argmax-abs-diff
//! always sampled first).
//!
//! Demonstrates the **ROSETTA-DIFF.2** recipe for PMAT-097 (apr rosetta diff-tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-188 + deterministic-sampling convention
//!
//! Run with: cargo run --example cli_rosetta_diff_tensors_value_sampler
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct SamplePair {
    pub index: usize,
    pub ref_value: f64,
    pub test_value: f64,
    pub abs_diff: f64,
}

pub fn sample_value_pairs(reference: &[f64], test: &[f64], n: usize) -> Vec<SamplePair> {
    if reference.len() != test.len() || reference.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut diffs: Vec<SamplePair> = reference
        .iter()
        .zip(test)
        .enumerate()
        .map(|(index, (r, t))| SamplePair {
            index,
            ref_value: *r,
            test_value: *t,
            abs_diff: (r - t).abs(),
        })
        .collect();

    // Always include the argmax-abs-diff first (most informative pair).
    diffs.sort_by(|a, b| {
        b.abs_diff
            .partial_cmp(&a.abs_diff)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    diffs.truncate(n);

    // Re-sort by index so the output is deterministic per (reference, test, n).
    diffs.sort_by_key(|p| p.index);
    diffs
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_diff_tensors_value_sampler")?;

    let r: Vec<f64> = (0..16).map(f64::from).collect();
    let mut t = r.clone();
    t[3] = 99.0; // big difference
    t[7] = -50.0; // negative big difference
    t[11] += 0.0001; // tiny

    println!("=== Recipe: cli_rosetta_diff_tensors_value_sampler ===");
    for n in [3, 5, 16, 0] {
        println!("--show-values {n}:");
        for sample in sample_value_pairs(&r, &t, n) {
            println!(
                "  [{:>2}]  ref={:>6.2}  test={:>6.2}  |diff|={:.4}",
                sample.index, sample.ref_value, sample.test_value, sample.abs_diff
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_inputs_yield_empty_samples() {
        assert!(sample_value_pairs(&[], &[], 5).is_empty());
    }

    #[test]
    fn shape_mismatch_yields_empty_samples() {
        // Shape mismatch is reported elsewhere; sampler returns nothing.
        assert!(sample_value_pairs(&[1.0, 2.0], &[1.0], 5).is_empty());
    }

    #[test]
    fn n_zero_yields_empty_samples() {
        assert!(sample_value_pairs(&[1.0, 2.0], &[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn always_includes_argmax_abs_diff() {
        // The most-divergent pair must appear in the output regardless of n.
        let r = vec![0.0; 10];
        let mut t = r.clone();
        t[7] = 100.0; // big spike
        let samples = sample_value_pairs(&r, &t, 1);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].index, 7);
        assert_eq!(samples[0].abs_diff, 100.0);
    }

    #[test]
    fn output_is_deterministic_for_same_inputs() {
        let r: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let t: Vec<f64> = (0..32).map(|i| (i as f64).cos()).collect();
        assert_eq!(sample_value_pairs(&r, &t, 5), sample_value_pairs(&r, &t, 5));
    }

    #[test]
    fn output_indices_are_strictly_increasing() {
        // After re-sort by index, output must be a strict ascending sequence.
        let r = vec![0.0; 10];
        let mut t = r.clone();
        t[1] = 5.0;
        t[3] = 10.0;
        t[8] = 20.0;
        let samples = sample_value_pairs(&r, &t, 3);
        for w in samples.windows(2) {
            assert!(w[1].index > w[0].index);
        }
    }

    #[test]
    fn n_larger_than_input_returns_full_input() {
        let r = vec![1.0, 2.0, 3.0];
        let t = vec![1.5, 2.5, 3.5];
        let samples = sample_value_pairs(&r, &t, 100);
        assert_eq!(samples.len(), 3);
    }
}
