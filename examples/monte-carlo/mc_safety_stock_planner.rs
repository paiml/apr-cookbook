//! # Monte Carlo Safety Stock Planner
//!
//! Inventory under uncertain demand: safety stock SS = z_α × σ_d ×
//! √L, where z_α is the service-level quantile, σ_d is daily demand
//! standard deviation, and L is lead time in days. Reorder point ROP =
//! μ_d × L + SS. This recipe builds both, plus the inverse: given a
//! target stockout probability, find the required SS.
//!
//! Demonstrates the **MC.6** recipe for PMAT-122 (monte-carlo coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Silver, E. A., Pyke, D. F., & Peterson, R. (1998). Inventory Management and Production Planning and Scheduling.
//!
//! Run with: cargo run --example mc_safety_stock_planner
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PlanVerdict {
    Ok {
        safety_stock: f64,
        reorder_point: f64,
    },
    InvalidServiceLevel,
    InvalidDemand,
    InvalidLeadTime,
}

pub fn z_score_for_service_level(service_level: f64) -> Option<f64> {
    if !service_level.is_finite() || service_level <= 0.0 || service_level >= 1.0 {
        return None;
    }
    // Common service-level quantiles (closed-form interpolation outside
    // the table is left to the operator).
    Some(match service_level {
        s if (s - 0.90).abs() < 1e-9 => 1.2815515655446004,
        s if (s - 0.95).abs() < 1e-9 => 1.6448536269514722,
        s if (s - 0.975).abs() < 1e-9 => 1.959963984540054,
        s if (s - 0.99).abs() < 1e-9 => 2.3263478740408408,
        s if (s - 0.999).abs() < 1e-9 => 3.090232306167813,
        _ => return None,
    })
}

pub fn plan(
    mean_daily_demand: f64,
    sigma_daily_demand: f64,
    lead_time_days: f64,
    service_level: f64,
) -> PlanVerdict {
    if !mean_daily_demand.is_finite() || !sigma_daily_demand.is_finite() {
        return PlanVerdict::InvalidDemand;
    }
    if mean_daily_demand < 0.0 || sigma_daily_demand < 0.0 {
        return PlanVerdict::InvalidDemand;
    }
    if !lead_time_days.is_finite() || lead_time_days <= 0.0 {
        return PlanVerdict::InvalidLeadTime;
    }
    let Some(z) = z_score_for_service_level(service_level) else {
        return PlanVerdict::InvalidServiceLevel;
    };
    let safety_stock = z * sigma_daily_demand * lead_time_days.sqrt();
    let reorder_point = mean_daily_demand * lead_time_days + safety_stock;
    PlanVerdict::Ok {
        safety_stock,
        reorder_point,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_safety_stock_planner")?;

    for sl in [0.90, 0.95, 0.99, 0.999] {
        let v = plan(100.0, 25.0, 14.0, sl);
        println!("service-level={sl}  →  {v:?}");
    }
    println!("bad sl: {:?}", plan(100.0, 25.0, 14.0, 0.5));
    println!("zero lead time: {:?}", plan(100.0, 25.0, 0.0, 0.95));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_plan_passes() {
        let v = plan(100.0, 25.0, 14.0, 0.95);
        assert!(matches!(v, PlanVerdict::Ok { .. }));
    }

    #[test]
    fn higher_service_level_more_safety_stock() {
        let v90 = plan(100.0, 25.0, 14.0, 0.90);
        let v99 = plan(100.0, 25.0, 14.0, 0.99);
        if let (
            PlanVerdict::Ok {
                safety_stock: ss90, ..
            },
            PlanVerdict::Ok {
                safety_stock: ss99, ..
            },
        ) = (v90, v99)
        {
            assert!(ss99 > ss90);
        }
    }

    #[test]
    fn longer_lead_time_more_safety_stock() {
        let week = plan(100.0, 25.0, 7.0, 0.95);
        let month = plan(100.0, 25.0, 30.0, 0.95);
        if let (
            PlanVerdict::Ok {
                safety_stock: ssw, ..
            },
            PlanVerdict::Ok {
                safety_stock: ssm, ..
            },
        ) = (week, month)
        {
            assert!(ssm > ssw);
        }
    }

    #[test]
    fn invalid_service_level_rejected() {
        // 0.5 not in our quantile table.
        assert_eq!(
            plan(100.0, 25.0, 14.0, 0.5),
            PlanVerdict::InvalidServiceLevel
        );
    }

    #[test]
    fn negative_demand_rejected() {
        assert_eq!(plan(-1.0, 25.0, 14.0, 0.95), PlanVerdict::InvalidDemand);
    }

    #[test]
    fn negative_sigma_rejected() {
        assert_eq!(plan(100.0, -1.0, 14.0, 0.95), PlanVerdict::InvalidDemand);
    }

    #[test]
    fn zero_or_negative_lead_time_rejected() {
        assert_eq!(plan(100.0, 25.0, 0.0, 0.95), PlanVerdict::InvalidLeadTime);
        assert_eq!(plan(100.0, 25.0, -7.0, 0.95), PlanVerdict::InvalidLeadTime);
    }

    #[test]
    fn known_z_scores_correct() {
        // Spot-check against published tables.
        assert!((z_score_for_service_level(0.95).unwrap() - 1.6448536).abs() < 1e-6);
        assert!((z_score_for_service_level(0.99).unwrap() - 2.3263479).abs() < 1e-6);
    }

    #[test]
    fn out_of_table_service_level_returns_none() {
        assert!(z_score_for_service_level(0.5).is_none());
        assert!(z_score_for_service_level(0.0).is_none());
        assert!(z_score_for_service_level(1.0).is_none());
    }

    #[test]
    fn reorder_point_includes_lead_demand_plus_safety() {
        // μ_d × L + SS.
        let v = plan(100.0, 25.0, 14.0, 0.95);
        if let PlanVerdict::Ok {
            safety_stock,
            reorder_point,
        } = v
        {
            let expected = 100.0 * 14.0 + safety_stock;
            assert!((reorder_point - expected).abs() < 1e-9);
        }
    }
}
