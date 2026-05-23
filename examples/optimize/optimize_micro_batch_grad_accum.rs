//! # Optimize Micro-Batch + Gradient Accumulation Sizer
//!
//! For a target effective batch B and per-device microbatch m, the
//! gradient accumulation steps k = B / (m × num_devices) must be a
//! positive integer. This recipe finds compatible (m, k) pairs for a
//! target B, given device count and per-device memory cap.
//!
//! Demonstrates the **OPT.26** recipe for PMAT-131 (optimize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA NeMo § Gradient Accumulation.
//!
//! Run with: cargo run --example optimize_micro_batch_grad_accum
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PlanEntry {
    pub micro_batch: u32,
    pub accum_steps: u32,
}

#[derive(Debug, PartialEq)]
pub enum SizerVerdict {
    Ok { plans: Vec<PlanEntry> },
    NoCompatibleSplit { effective: u32, devices: u32 },
    InvalidEffective,
    InvalidDevices,
    InvalidMicroBatchCap,
}

pub fn enumerate(effective_batch: u32, num_devices: u32, micro_batch_cap: u32) -> SizerVerdict {
    if effective_batch == 0 {
        return SizerVerdict::InvalidEffective;
    }
    if num_devices == 0 {
        return SizerVerdict::InvalidDevices;
    }
    if micro_batch_cap == 0 {
        return SizerVerdict::InvalidMicroBatchCap;
    }
    let parallel = num_devices;
    let mut plans = Vec::new();
    for m in 1..=micro_batch_cap {
        let prod = m * parallel;
        if effective_batch % prod != 0 {
            continue;
        }
        let k = effective_batch / prod;
        if k > 0 {
            plans.push(PlanEntry {
                micro_batch: m,
                accum_steps: k,
            });
        }
    }
    if plans.is_empty() {
        return SizerVerdict::NoCompatibleSplit {
            effective: effective_batch,
            devices: num_devices,
        };
    }
    SizerVerdict::Ok { plans }
}

pub fn pick_largest_micro(plans: &[PlanEntry]) -> Option<PlanEntry> {
    plans.iter().max_by_key(|p| p.micro_batch).copied()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("optimize_micro_batch_grad_accum")?;

    println!("256 / 8 dev / cap 32: {:?}", enumerate(256, 8, 32));
    println!("251 / 8 dev / cap 32: {:?}", enumerate(251, 8, 32));
    if let SizerVerdict::Ok { plans } = enumerate(256, 8, 32) {
        println!("largest m: {:?}", pick_largest_micro(&plans));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_batch_split_enumerated() {
        if let SizerVerdict::Ok { plans } = enumerate(256, 8, 32) {
            // Possible (m, k): (1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1).
            assert_eq!(plans.len(), 6);
            assert!(plans
                .iter()
                .any(|p| p.micro_batch == 32 && p.accum_steps == 1));
            assert!(plans
                .iter()
                .any(|p| p.micro_batch == 1 && p.accum_steps == 32));
        }
    }

    #[test]
    fn prime_effective_batch_no_split() {
        let v = enumerate(251, 8, 32);
        assert!(matches!(v, SizerVerdict::NoCompatibleSplit { .. }));
    }

    #[test]
    fn cap_excludes_large_microbatch() {
        if let SizerVerdict::Ok { plans } = enumerate(256, 8, 16) {
            // Largest m allowed is 16.
            assert!(plans.iter().all(|p| p.micro_batch <= 16));
        }
    }

    #[test]
    fn invalid_effective_rejected() {
        assert_eq!(enumerate(0, 8, 32), SizerVerdict::InvalidEffective);
    }

    #[test]
    fn invalid_devices_rejected() {
        assert_eq!(enumerate(256, 0, 32), SizerVerdict::InvalidDevices);
    }

    #[test]
    fn invalid_cap_rejected() {
        assert_eq!(enumerate(256, 8, 0), SizerVerdict::InvalidMicroBatchCap);
    }

    #[test]
    fn single_device_gives_full_range() {
        if let SizerVerdict::Ok { plans } = enumerate(8, 1, 32) {
            // Possible (m, k): (1, 8), (2, 4), (4, 2), (8, 1).
            assert_eq!(plans.len(), 4);
        }
    }

    #[test]
    fn pick_largest_returns_max_m() {
        if let SizerVerdict::Ok { plans } = enumerate(256, 8, 32) {
            let largest = pick_largest_micro(&plans).unwrap();
            assert_eq!(largest.micro_batch, 32);
            assert_eq!(largest.accum_steps, 1);
        }
    }

    #[test]
    fn pick_largest_empty_yields_none() {
        assert!(pick_largest_micro(&[]).is_none());
    }

    #[test]
    fn product_invariant_holds() {
        if let SizerVerdict::Ok { plans } = enumerate(256, 8, 32) {
            for p in plans {
                assert_eq!(p.micro_batch * 8 * p.accum_steps, 256);
            }
        }
    }
}
