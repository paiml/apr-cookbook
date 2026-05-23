//! # Monitoring Population Stability Index (PSI) Drift
//!
//! PSI quantifies distribution drift between baseline and current
//! sample bins:
//!   PSI = Σ (current_pct - baseline_pct) × ln(current_pct / baseline_pct)
//!
//! Tier guideline:
//!   < 0.10  → no significant drift
//!   0.10-0.25 → moderate drift, investigate
//!   ≥ 0.25  → significant drift, retrain candidate
//!
//! This recipe builds the calculator + tier picker.
//!
//! Demonstrates the **MON.16** recipe for PMAT-137 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PSI methodology (credit-scoring industry standard).
//!
//! Run with: cargo run --example monitor_drift_psi
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SMOOTHING: f64 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftTier {
    NoDrift,
    Moderate,
    Significant,
}

#[derive(Debug, PartialEq)]
pub enum PsiVerdict {
    Ok { psi: f64, tier: DriftTier },
    BinCountMismatch { baseline: usize, current: usize },
    EmptyBins,
    NegativeCounts,
}

pub fn calculate(baseline: &[u32], current: &[u32]) -> PsiVerdict {
    if baseline.len() != current.len() {
        return PsiVerdict::BinCountMismatch {
            baseline: baseline.len(),
            current: current.len(),
        };
    }
    if baseline.is_empty() {
        return PsiVerdict::EmptyBins;
    }
    let baseline_total: u64 = baseline.iter().map(|x| u64::from(*x)).sum();
    let current_total: u64 = current.iter().map(|x| u64::from(*x)).sum();
    if baseline_total == 0 || current_total == 0 {
        return PsiVerdict::NegativeCounts;
    }
    let mut psi = 0.0f64;
    for (b, c) in baseline.iter().zip(current.iter()) {
        let bp = (f64::from(*b) / baseline_total as f64).max(SMOOTHING);
        let cp = (f64::from(*c) / current_total as f64).max(SMOOTHING);
        psi += (cp - bp) * (cp / bp).ln();
    }
    let tier = if psi < 0.10 {
        DriftTier::NoDrift
    } else if psi < 0.25 {
        DriftTier::Moderate
    } else {
        DriftTier::Significant
    };
    PsiVerdict::Ok { psi, tier }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_drift_psi")?;

    let baseline = [100u32, 200, 300, 200, 100];
    println!("identical: {:?}", calculate(&baseline, &baseline));
    println!(
        "mild shift: {:?}",
        calculate(&baseline, &[110, 210, 290, 200, 90])
    );
    println!(
        "significant shift: {:?}",
        calculate(&baseline, &[200, 300, 100, 50, 50])
    );
    println!(
        "size mismatch: {:?}",
        calculate(&baseline, &[100, 200, 300])
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_distributions_zero_psi() {
        let baseline = [100u32, 200, 300, 200, 100];
        if let PsiVerdict::Ok { psi, tier } = calculate(&baseline, &baseline) {
            assert!(psi.abs() < 1e-9);
            assert_eq!(tier, DriftTier::NoDrift);
        }
    }

    #[test]
    fn mild_shift_no_drift_tier() {
        let baseline = [100u32, 200, 300, 200, 100];
        let current = [110u32, 210, 290, 200, 90];
        if let PsiVerdict::Ok { tier, .. } = calculate(&baseline, &current) {
            assert_eq!(tier, DriftTier::NoDrift);
        }
    }

    #[test]
    fn significant_shift_significant_tier() {
        let baseline = [100u32, 200, 300, 200, 100];
        let current = [600u32, 100, 50, 25, 25];
        if let PsiVerdict::Ok { tier, .. } = calculate(&baseline, &current) {
            assert_eq!(tier, DriftTier::Significant);
        }
    }

    #[test]
    fn size_mismatch_rejected() {
        let v = calculate(&[100, 200, 300], &[100, 200]);
        assert!(matches!(v, PsiVerdict::BinCountMismatch { .. }));
    }

    #[test]
    fn empty_bins_rejected() {
        assert_eq!(calculate(&[], &[]), PsiVerdict::EmptyBins);
    }

    #[test]
    fn zero_total_rejected() {
        assert_eq!(
            calculate(&[0, 0, 0], &[100, 200, 300]),
            PsiVerdict::NegativeCounts
        );
    }

    #[test]
    fn psi_non_negative() {
        // PSI is always >= 0 (KL-divergence-like, symmetric formulation here).
        let baseline = [100u32, 200, 300];
        let current = [50u32, 100, 800];
        if let PsiVerdict::Ok { psi, .. } = calculate(&baseline, &current) {
            assert!(psi >= 0.0);
        }
    }

    #[test]
    fn moderate_shift_moderate_tier() {
        // Tuned values to land in 0.10..0.25 PSI range.
        let baseline = [100u32, 100, 100, 100, 100];
        let current = [180u32, 100, 100, 60, 60];
        if let PsiVerdict::Ok { psi, tier } = calculate(&baseline, &current) {
            // Confirm in moderate band.
            assert!(psi >= 0.10, "psi={psi} < 0.10");
            assert!(psi < 0.25, "psi={psi} >= 0.25");
            assert_eq!(tier, DriftTier::Moderate);
        }
    }

    #[test]
    fn smoothing_handles_empty_bin() {
        // Current has zero count in one bin; smoothing prevents NaN.
        let baseline = [100u32, 100, 100];
        let current = [0u32, 100, 100];
        if let PsiVerdict::Ok { psi, .. } = calculate(&baseline, &current) {
            assert!(psi.is_finite());
        }
    }

    #[test]
    fn symmetric_under_swap_for_balanced() {
        // PSI uses (cur - base)(ln cur/base) which is symmetric in base ↔ cur.
        let baseline = [100u32, 200, 300];
        let current = [300u32, 200, 100];
        let psi_a = if let PsiVerdict::Ok { psi, .. } = calculate(&baseline, &current) {
            psi
        } else {
            f64::NAN
        };
        let psi_b = if let PsiVerdict::Ok { psi, .. } = calculate(&current, &baseline) {
            psi
        } else {
            f64::NAN
        };
        assert!((psi_a - psi_b).abs() < 1e-9);
    }
}
