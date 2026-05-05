//! # apr diff --values — Element-wise APRT Stage Diff
//!
//! `apr diff --values` recognizes APRT stage tensors (per aprender PR #1413)
//! so you can do element-wise CPU/GPU bisection without round-tripping
//! through SafeTensors. This recipe demonstrates the cosine-similarity +
//! max-absolute-error metrics the real `apr diff --values` reports for two
//! APRT files at the same forward-pass stage.
//!
//! Demonstrates the **CLI+.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PR #1413 + Manning, Raghavan, Schütze (2008). Introduction to Information Retrieval. Cambridge UP. ISBN: 978-0521865715
//!
//! Run with: cargo run --example cli_diff_values_aprt_stage
//!
//! Added by PMAT-075 (expand-cookbooks: GPU/CPU oracle bisection).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug)]
struct DiffReport {
    cosine_similarity: f64,
    max_abs_error: f64,
    mean_abs_error: f64,
    n_elements: usize,
}

fn diff_tensors(a: &[f32], b: &[f32]) -> Result<DiffReport> {
    if a.len() != b.len() {
        return Err(apr_cookbook::CookbookError::Validation(format!(
            "tensor length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(apr_cookbook::CookbookError::Validation(
            "empty tensors cannot be diffed".into(),
        ));
    }
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cosine_similarity = if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    };
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64 - *y as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
    }
    Ok(DiffReport {
        cosine_similarity,
        max_abs_error: max_abs,
        mean_abs_error: sum_abs / a.len() as f64,
        n_elements: a.len(),
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diff_values_aprt_stage")?;

    // Synthetic CPU vs GPU layer-0 attn_norm output — close but not identical.
    let cpu: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let gpu: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 0.0001).collect();

    let report = diff_tensors(&cpu, &gpu)?;
    println!("APRT stage diff: layer-0 attn_norm (CPU vs GPU)");
    println!("  n_elements:        {}", report.n_elements);
    println!("  cosine_similarity: {:.6}", report.cosine_similarity);
    println!("  max_abs_error:     {:.6}", report.max_abs_error);
    println!("  mean_abs_error:    {:.6}", report.mean_abs_error);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tensors_yield_cosine_one() {
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let report = diff_tensors(&v, &v).unwrap();
        assert!((report.cosine_similarity - 1.0).abs() < 1e-12);
        assert!(report.max_abs_error < 1e-12);
    }

    #[test]
    fn length_mismatch_rejected() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert!(diff_tensors(&a, &b).is_err());
    }

    #[test]
    fn empty_tensors_rejected() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(diff_tensors(&a, &b).is_err());
    }

    #[test]
    fn small_perturbation_keeps_cosine_close_to_one() {
        let cpu: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let gpu: Vec<f32> = cpu.iter().map(|x| x + 0.0001).collect();
        let report = diff_tensors(&cpu, &gpu).unwrap();
        assert!(
            report.cosine_similarity > 0.9999,
            "expected cosine>0.9999 for tiny perturbation, got {:.6}",
            report.cosine_similarity
        );
    }
}
