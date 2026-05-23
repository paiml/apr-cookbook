//! # Advanced Request Priority Aging
//!
//! Anti-starvation: bump a request's effective priority based on time
//! waited. Effective priority = base_priority - (waited_secs / step_secs)
//! clamped to 0 (highest). Lower numeric = higher priority.
//!
//! Demonstrates the **ADV.40** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Operating-system aging (Tanenbaum, Modern OS).
//!
//! Run with: cargo run --example adv_request_priority_aging
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AgeVerdict {
    Ok { effective_priority: u8, bumps: u32 },
    InvalidConfig,
}

pub fn age(base_priority: u8, waited_secs: u64, step_secs: u64) -> AgeVerdict {
    if step_secs == 0 {
        return AgeVerdict::InvalidConfig;
    }
    let bumps = (waited_secs / step_secs) as u32;
    let effective_priority = base_priority.saturating_sub(bumps as u8);
    AgeVerdict::Ok {
        effective_priority,
        bumps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_request_priority_aging")?;

    println!("no wait: {:?}", age(10, 0, 60));
    println!("waited 5 min: {:?}", age(10, 300, 60));
    println!("very long: {:?}", age(10, 7200, 60));
    println!("invalid: {:?}", age(10, 60, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ager_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_wait_no_bumps() {
        let v = age(10, 0, 60);
        if let AgeVerdict::Ok {
            effective_priority,
            bumps,
        } = v
        {
            assert_eq!(effective_priority, 10);
            assert_eq!(bumps, 0);
        }
    }

    #[test]
    fn five_minutes_5_bumps() {
        let v = age(10, 300, 60);
        if let AgeVerdict::Ok {
            effective_priority,
            bumps,
        } = v
        {
            assert_eq!(bumps, 5);
            assert_eq!(effective_priority, 5);
        }
    }

    #[test]
    fn extremely_long_clamped_to_zero() {
        let v = age(10, 7200, 60);
        if let AgeVerdict::Ok {
            effective_priority, ..
        } = v
        {
            // Saturating sub at 0.
            assert_eq!(effective_priority, 0);
        }
    }

    #[test]
    fn zero_step_invalid() {
        assert_eq!(age(10, 60, 0), AgeVerdict::InvalidConfig);
    }

    #[test]
    fn just_under_step_no_bump() {
        let v = age(10, 59, 60);
        if let AgeVerdict::Ok { bumps, .. } = v {
            assert_eq!(bumps, 0);
        }
    }

    #[test]
    fn exactly_one_step_one_bump() {
        let v = age(10, 60, 60);
        if let AgeVerdict::Ok { bumps, .. } = v {
            assert_eq!(bumps, 1);
        }
    }

    #[test]
    fn effective_priority_never_overflows() {
        let v = age(255, 0, 1);
        if let AgeVerdict::Ok {
            effective_priority, ..
        } = v
        {
            assert_eq!(effective_priority, 255);
        }
    }

    #[test]
    fn priority_zero_unchanged() {
        let v = age(0, 1000, 60);
        if let AgeVerdict::Ok {
            effective_priority, ..
        } = v
        {
            assert_eq!(effective_priority, 0);
        }
    }

    #[test]
    fn deterministic() {
        let a = age(10, 300, 60);
        let b = age(10, 300, 60);
        assert_eq!(a, b);
    }

    #[test]
    fn bumps_value_independent_of_priority() {
        let a = age(255, 600, 60);
        let b = age(0, 600, 60);
        if let (AgeVerdict::Ok { bumps: ba, .. }, AgeVerdict::Ok { bumps: bb, .. }) = (a, b) {
            assert_eq!(ba, bb);
            assert_eq!(ba, 10);
        }
    }
}
