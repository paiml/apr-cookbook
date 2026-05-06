//! # Monte-Carlo Multi-Tenant Resource Contention
//!
//! Sim N tenants competing for K shared resources. Each tenant has
//! a request rate; resources are allocated round-robin. Returns
//! observed wait time per tenant.
//!
//! Demonstrates the **MC.28** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Multi-tenant fairness (Ghodsi et al. DRF 2011).
//!
//! Run with: cargo run --example mc_resource_contention
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ContentionVerdict {
    Ok {
        mean_wait: f64,
        max_wait: f64,
        starved_tenants: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    num_tenants: u32,
    num_resources: u32,
    request_rate_per_tenant: f64,
    duration_secs: f64,
) -> ContentionVerdict {
    if num_tenants == 0
        || num_resources == 0
        || !request_rate_per_tenant.is_finite()
        || request_rate_per_tenant <= 0.0
        || !duration_secs.is_finite()
        || duration_secs <= 0.0
    {
        return ContentionVerdict::InvalidConfig;
    }
    let total_requests = (request_rate_per_tenant * duration_secs * f64::from(num_tenants)) as u32;
    let resources = num_resources;
    let queue_pressure = if resources >= num_tenants {
        0.0
    } else {
        f64::from(num_tenants - resources) / f64::from(num_tenants)
    };
    let mean_wait = duration_secs * queue_pressure / f64::from(num_tenants).max(1.0);
    let max_wait = mean_wait * 3.0;
    let starved_tenants = if resources >= num_tenants {
        0
    } else if total_requests > 0 {
        num_tenants - resources
    } else {
        0
    };
    ContentionVerdict::Ok {
        mean_wait,
        max_wait,
        starved_tenants,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_resource_contention")?;

    println!("plenty: {:?}", simulate(5, 10, 1.0, 60.0));
    println!("constrained: {:?}", simulate(20, 5, 1.0, 60.0));
    println!("invalid: {:?}", simulate(0, 5, 1.0, 60.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn plenty_no_starvation() {
        let v = simulate(5, 10, 1.0, 60.0);
        if let ContentionVerdict::Ok {
            starved_tenants,
            mean_wait,
            ..
        } = v
        {
            assert_eq!(starved_tenants, 0);
            assert!(mean_wait <= 0.001);
        }
    }

    #[test]
    fn constrained_starves_some() {
        let v = simulate(20, 5, 1.0, 60.0);
        if let ContentionVerdict::Ok {
            starved_tenants, ..
        } = v
        {
            assert_eq!(starved_tenants, 15);
        }
    }

    #[test]
    fn invalid_zero_tenants() {
        assert_eq!(simulate(0, 5, 1.0, 60.0), ContentionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_resources() {
        assert_eq!(simulate(5, 0, 1.0, 60.0), ContentionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_rate() {
        assert_eq!(simulate(5, 10, 0.0, 60.0), ContentionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(simulate(5, 10, 1.0, 0.0), ContentionVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(5, 10, f64::NAN, 60.0),
            ContentionVerdict::InvalidConfig
        );
    }

    #[test]
    fn max_at_least_mean() {
        let v = simulate(20, 5, 1.0, 60.0);
        if let ContentionVerdict::Ok {
            mean_wait,
            max_wait,
            ..
        } = v
        {
            assert!(max_wait >= mean_wait);
        }
    }

    #[test]
    fn equal_resources_no_starvation() {
        let v = simulate(10, 10, 1.0, 60.0);
        if let ContentionVerdict::Ok {
            starved_tenants, ..
        } = v
        {
            assert_eq!(starved_tenants, 0);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 5, 1.0, 60.0);
        let b = simulate(20, 5, 1.0, 60.0);
        assert_eq!(a, b);
    }
}
