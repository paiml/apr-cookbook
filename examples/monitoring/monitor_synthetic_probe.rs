//! # Monitoring Synthetic Probe Frequency Picker
//!
//! How often to ping a service for liveness:
//!   internal critical service: 10s (catches outage fast)
//!   user-facing API: 30s
//!   batch / async: 5min
//!   external dependency: 1min
//!
//! Picker by service tier, returns frequency + alarm threshold (N
//! consecutive failures before paging).
//!
//! Demonstrates the **MON.33** recipe for PMAT-151 (monitoring round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Datadog/Pingdom synthetic probe best practices.
//!
//! Run with: cargo run --example monitor_synthetic_probe
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    InternalCritical,
    UserFacingApi,
    BatchAsync,
    ExternalDep,
}

#[derive(Debug, PartialEq)]
pub enum ProbeVerdict {
    Ok {
        frequency_secs: u32,
        consecutive_failures_to_page: u32,
    },
    InvalidTier,
}

pub fn pick(tier: ServiceTier) -> ProbeVerdict {
    let (frequency_secs, consecutive_failures_to_page) = match tier {
        ServiceTier::InternalCritical => (10, 2),
        ServiceTier::UserFacingApi => (30, 3),
        ServiceTier::ExternalDep => (60, 5),
        ServiceTier::BatchAsync => (300, 2),
    };
    ProbeVerdict::Ok {
        frequency_secs,
        consecutive_failures_to_page,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_synthetic_probe")?;

    for tier in [
        ServiceTier::InternalCritical,
        ServiceTier::UserFacingApi,
        ServiceTier::BatchAsync,
        ServiceTier::ExternalDep,
    ] {
        println!("{tier:?}: {:?}", pick(tier));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn internal_critical_frequent() {
        let v = pick(ServiceTier::InternalCritical);
        if let ProbeVerdict::Ok { frequency_secs, .. } = v {
            assert_eq!(frequency_secs, 10);
        }
    }

    #[test]
    fn user_api_frequency_30s() {
        let v = pick(ServiceTier::UserFacingApi);
        if let ProbeVerdict::Ok { frequency_secs, .. } = v {
            assert_eq!(frequency_secs, 30);
        }
    }

    #[test]
    fn external_one_minute() {
        let v = pick(ServiceTier::ExternalDep);
        if let ProbeVerdict::Ok { frequency_secs, .. } = v {
            assert_eq!(frequency_secs, 60);
        }
    }

    #[test]
    fn batch_async_low_frequency() {
        let v = pick(ServiceTier::BatchAsync);
        if let ProbeVerdict::Ok { frequency_secs, .. } = v {
            assert_eq!(frequency_secs, 300);
        }
    }

    #[test]
    fn critical_pages_after_2_failures() {
        let v = pick(ServiceTier::InternalCritical);
        if let ProbeVerdict::Ok {
            consecutive_failures_to_page,
            ..
        } = v
        {
            assert_eq!(consecutive_failures_to_page, 2);
        }
    }

    #[test]
    fn external_pages_after_more_failures() {
        // External APIs flap; need more failures to be confident.
        let v = pick(ServiceTier::ExternalDep);
        if let ProbeVerdict::Ok {
            consecutive_failures_to_page,
            ..
        } = v
        {
            assert_eq!(consecutive_failures_to_page, 5);
        }
    }

    #[test]
    fn critical_higher_frequency_than_batch() {
        let v_crit = pick(ServiceTier::InternalCritical);
        let v_batch = pick(ServiceTier::BatchAsync);
        if let (
            ProbeVerdict::Ok {
                frequency_secs: c, ..
            },
            ProbeVerdict::Ok {
                frequency_secs: b, ..
            },
        ) = (v_crit, v_batch)
        {
            assert!(c < b);
        }
    }

    #[test]
    fn external_more_failures_than_internal() {
        let v_int = pick(ServiceTier::InternalCritical);
        let v_ext = pick(ServiceTier::ExternalDep);
        if let (
            ProbeVerdict::Ok {
                consecutive_failures_to_page: i,
                ..
            },
            ProbeVerdict::Ok {
                consecutive_failures_to_page: e,
                ..
            },
        ) = (v_int, v_ext)
        {
            assert!(e > i);
        }
    }

    #[test]
    fn ok_for_all_tiers() {
        for tier in [
            ServiceTier::InternalCritical,
            ServiceTier::UserFacingApi,
            ServiceTier::BatchAsync,
            ServiceTier::ExternalDep,
        ] {
            assert!(matches!(pick(tier), ProbeVerdict::Ok { .. }));
        }
    }

    #[test]
    fn deterministic_per_tier() {
        let a = pick(ServiceTier::UserFacingApi);
        let b = pick(ServiceTier::UserFacingApi);
        assert_eq!(a, b);
    }

    #[test]
    fn pages_never_zero() {
        for tier in [
            ServiceTier::InternalCritical,
            ServiceTier::UserFacingApi,
            ServiceTier::BatchAsync,
            ServiceTier::ExternalDep,
        ] {
            if let ProbeVerdict::Ok {
                consecutive_failures_to_page,
                ..
            } = pick(tier)
            {
                assert!(consecutive_failures_to_page >= 1);
            }
        }
    }
}
