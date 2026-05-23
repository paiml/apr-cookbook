//! # Monitoring SLO Burn-Rate Alerter
//!
//! Google SRE multi-window burn-rate alert: combine a long window
//! (1h, slow burn) and a short window (5m, fast burn) of error-rate.
//! Burn rate = error_rate / (1 - SLO). Alert thresholds:
//!   1h × 14.4×  AND 5m × 14.4× → page (will exhaust budget in 1d)
//!   6h × 6×    AND 30m × 6×   → page (3.5d budget)
//!   1d × 3×    AND 2h × 3×    → ticket (10d)
//!   3d × 1×    AND 6h × 1×    → ticket (30d)
//!
//! This recipe builds the threshold checker.
//!
//! Demonstrates the **MON.18** recipe for PMAT-137 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Google SRE Workbook ch.5 — Alerting on SLOs.
//!
//! Run with: cargo run --example monitor_slo_burn_rate
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    None,
    Ticket,
    Page,
}

#[derive(Debug, PartialEq)]
pub enum SloVerdict {
    Ok {
        severity: AlertSeverity,
        long_burn_rate: f64,
        short_burn_rate: f64,
    },
    InvalidSlo,
    InvalidErrorRate,
}

pub fn evaluate(
    slo_target: f64,
    long_window_error_rate: f64,
    short_window_error_rate: f64,
) -> SloVerdict {
    if !slo_target.is_finite() || slo_target <= 0.0 || slo_target >= 1.0 {
        return SloVerdict::InvalidSlo;
    }
    if !long_window_error_rate.is_finite()
        || !short_window_error_rate.is_finite()
        || long_window_error_rate < 0.0
        || short_window_error_rate < 0.0
    {
        return SloVerdict::InvalidErrorRate;
    }
    let allowed_error = 1.0 - slo_target;
    if allowed_error <= 0.0 {
        return SloVerdict::InvalidSlo;
    }
    let long = long_window_error_rate / allowed_error;
    let short = short_window_error_rate / allowed_error;
    let severity = if (long >= 14.4 && short >= 14.4) || (long >= 6.0 && short >= 6.0) {
        AlertSeverity::Page
    } else if (long >= 3.0 && short >= 3.0) || (long >= 1.0 && short >= 1.0) {
        AlertSeverity::Ticket
    } else {
        AlertSeverity::None
    };
    SloVerdict::Ok {
        severity,
        long_burn_rate: long,
        short_burn_rate: short,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_slo_burn_rate")?;

    let slo = 0.999;
    println!("normal: {:?}", evaluate(slo, 0.0005, 0.0005));
    println!("ticket-tier: {:?}", evaluate(slo, 0.005, 0.005));
    println!("page-tier 6×: {:?}", evaluate(slo, 0.007, 0.007));
    println!("page-tier 14.4×: {:?}", evaluate(slo, 0.02, 0.02));
    println!("invalid SLO: {:?}", evaluate(0.0, 0.001, 0.001));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alerter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_traffic_no_alert() {
        // 99.9% SLO, near-zero errors → severity None.
        let v = evaluate(0.999, 0.0001, 0.0001);
        if let SloVerdict::Ok { severity, .. } = v {
            assert_eq!(severity, AlertSeverity::None);
        }
    }

    #[test]
    fn high_burn_rate_pages() {
        // burn = 0.02 / 0.001 = 20 ≥ 14.4 in both → page.
        let v = evaluate(0.999, 0.02, 0.02);
        if let SloVerdict::Ok { severity, .. } = v {
            assert_eq!(severity, AlertSeverity::Page);
        }
    }

    #[test]
    fn medium_burn_rate_pages_at_six() {
        // burn = 0.007 / 0.001 = 7 ≥ 6 in both → page (tier 2).
        let v = evaluate(0.999, 0.007, 0.007);
        if let SloVerdict::Ok { severity, .. } = v {
            assert_eq!(severity, AlertSeverity::Page);
        }
    }

    #[test]
    fn low_burn_rate_tickets() {
        // burn = 0.002 / 0.001 = 2.0 → ticket tier.
        let v = evaluate(0.999, 0.002, 0.002);
        if let SloVerdict::Ok { severity, .. } = v {
            assert_eq!(severity, AlertSeverity::Ticket);
        }
    }

    #[test]
    fn slo_zero_invalid() {
        assert_eq!(evaluate(0.0, 0.001, 0.001), SloVerdict::InvalidSlo);
    }

    #[test]
    fn slo_at_one_invalid() {
        assert_eq!(evaluate(1.0, 0.001, 0.001), SloVerdict::InvalidSlo);
    }

    #[test]
    fn negative_error_invalid() {
        assert_eq!(evaluate(0.999, -0.001, 0.001), SloVerdict::InvalidErrorRate);
    }

    #[test]
    fn nan_error_invalid() {
        assert_eq!(
            evaluate(0.999, f64::NAN, 0.001),
            SloVerdict::InvalidErrorRate
        );
    }

    #[test]
    fn long_high_short_low_no_page() {
        // long burn high but short low → don't page yet (multi-window protects against false positives).
        let v = evaluate(0.999, 0.02, 0.0001);
        if let SloVerdict::Ok { severity, .. } = v {
            assert_ne!(severity, AlertSeverity::Page);
        }
    }

    #[test]
    fn burn_rates_reported() {
        // 0.005 / 0.001 = 5.
        if let SloVerdict::Ok { long_burn_rate, .. } = evaluate(0.999, 0.005, 0.005) {
            assert!((long_burn_rate - 5.0).abs() < 1e-9);
        }
    }
}
