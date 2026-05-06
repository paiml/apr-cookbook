//! # Monitoring SLA Uptime Calculator
//!
//! SLA targets: "three nines" = 99.9% (43.8 min downtime/month);
//! "four nines" = 99.99% (4.38 min/month); "five nines" = 99.999%
//! (26.3 sec/month). This recipe builds the calculator + tier
//! classifier + remaining-error-budget reporter.
//!
//! Demonstrates the **MON.9** recipe for PMAT-124 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Beyer et al. (2016). Site Reliability Engineering. O'Reilly.
//!
//! Run with: cargo run --example monitor_sla_uptime_calc
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub enum NinesTier {
    BelowTwoNines,
    TwoNines,
    ThreeNines,
    FourNines,
    FiveNines,
    SixNinesOrBetter,
    InvalidUptime,
}

pub fn classify_uptime(uptime_pct: f64) -> NinesTier {
    if !uptime_pct.is_finite() || !(0.0..=100.0).contains(&uptime_pct) {
        return NinesTier::InvalidUptime;
    }
    if uptime_pct >= 99.9999 {
        NinesTier::SixNinesOrBetter
    } else if uptime_pct >= 99.999 {
        NinesTier::FiveNines
    } else if uptime_pct >= 99.99 {
        NinesTier::FourNines
    } else if uptime_pct >= 99.9 {
        NinesTier::ThreeNines
    } else if uptime_pct >= 99.0 {
        NinesTier::TwoNines
    } else {
        NinesTier::BelowTwoNines
    }
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok { budget_seconds: f64 },
    InvalidWindow,
    InvalidTarget,
}

pub fn error_budget_seconds(target_uptime_pct: f64, window_days: u32) -> BudgetVerdict {
    if !target_uptime_pct.is_finite() || !(0.0..=100.0).contains(&target_uptime_pct) {
        return BudgetVerdict::InvalidTarget;
    }
    if window_days == 0 {
        return BudgetVerdict::InvalidWindow;
    }
    let downtime_pct = 100.0 - target_uptime_pct;
    let total_seconds = f64::from(window_days) * 86_400.0;
    BudgetVerdict::Ok {
        budget_seconds: total_seconds * downtime_pct / 100.0,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_sla_uptime_calc")?;

    for u in [99.0, 99.5, 99.9, 99.95, 99.99, 99.999, 100.0] {
        println!("uptime={u}%  →  {:?}", classify_uptime(u));
    }
    println!(
        "30-day budget @ 99.9%: {:?}",
        error_budget_seconds(99.9, 30)
    );
    println!(
        "90-day budget @ 99.99%: {:?}",
        error_budget_seconds(99.99, 90)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn three_nines_classified() {
        assert_eq!(classify_uptime(99.9), NinesTier::ThreeNines);
        assert_eq!(classify_uptime(99.95), NinesTier::ThreeNines);
    }

    #[test]
    fn four_nines_classified() {
        assert_eq!(classify_uptime(99.99), NinesTier::FourNines);
        assert_eq!(classify_uptime(99.995), NinesTier::FourNines);
    }

    #[test]
    fn five_nines_classified() {
        assert_eq!(classify_uptime(99.999), NinesTier::FiveNines);
    }

    #[test]
    fn six_nines_classified() {
        assert_eq!(classify_uptime(99.9999), NinesTier::SixNinesOrBetter);
        assert_eq!(classify_uptime(100.0), NinesTier::SixNinesOrBetter);
    }

    #[test]
    fn below_99_classified() {
        assert_eq!(classify_uptime(98.5), NinesTier::BelowTwoNines);
    }

    #[test]
    fn out_of_range_invalid() {
        assert_eq!(classify_uptime(-1.0), NinesTier::InvalidUptime);
        assert_eq!(classify_uptime(100.1), NinesTier::InvalidUptime);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(classify_uptime(f64::NAN), NinesTier::InvalidUptime);
    }

    #[test]
    fn three_nines_30_day_budget_43_min() {
        // 30 × 86400 × 0.001 = 2592 sec ≈ 43.2 min.
        if let BudgetVerdict::Ok { budget_seconds } = error_budget_seconds(99.9, 30) {
            assert!((budget_seconds - 2592.0).abs() < 1e-3);
        }
    }

    #[test]
    fn five_nines_30_day_budget_26_sec() {
        // 30 × 86400 × 0.00001 = 25.92 sec.
        if let BudgetVerdict::Ok { budget_seconds } = error_budget_seconds(99.999, 30) {
            assert!((budget_seconds - 25.92).abs() < 1e-3);
        }
    }

    #[test]
    fn zero_window_invalid() {
        assert_eq!(error_budget_seconds(99.9, 0), BudgetVerdict::InvalidWindow);
    }

    #[test]
    fn out_of_range_target_invalid() {
        assert_eq!(
            error_budget_seconds(110.0, 30),
            BudgetVerdict::InvalidTarget
        );
    }
}
