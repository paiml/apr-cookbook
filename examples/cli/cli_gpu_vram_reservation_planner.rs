//! # apr gpu — VRAM Reservation Planner
//!
//! `apr gpu` shows VRAM usage and reservations. This recipe builds the
//! reservation planner: given (total VRAM, current reservations,
//! requested allocation), decide whether the allocation fits without
//! evicting other tenants. Per GPU-SHARE-001, must keep ≥ 1 GB safety
//! margin.
//!
//! Demonstrates the **GPU.10** recipe for PMAT-107 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GPU-SHARE-001 + safety-margin convention
//!
//! Run with: cargo run --example cli_gpu_vram_reservation_planner
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SAFETY_MARGIN_BYTES: u64 = 1_000_000_000; // 1 GB

#[derive(Debug, Clone, PartialEq)]
pub struct Reservation {
    pub tenant: String,
    pub bytes: u64,
}

#[derive(Debug, PartialEq)]
pub enum AllocVerdict {
    Allocated,
    InsufficientFreeVram { available: u64, required: u64 },
    WouldViolateSafetyMargin,
    EmptyTenant,
}

pub fn try_allocate(
    total_vram: u64,
    existing: &[Reservation],
    new_tenant: &str,
    new_bytes: u64,
) -> AllocVerdict {
    if new_tenant.is_empty() {
        return AllocVerdict::EmptyTenant;
    }
    let used: u64 = existing.iter().map(|r| r.bytes).sum();
    let free = total_vram.saturating_sub(used);
    if new_bytes > free {
        return AllocVerdict::InsufficientFreeVram {
            available: free,
            required: new_bytes,
        };
    }
    let post_alloc_free = free - new_bytes;
    if post_alloc_free < SAFETY_MARGIN_BYTES {
        return AllocVerdict::WouldViolateSafetyMargin;
    }
    AllocVerdict::Allocated
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_vram_reservation_planner")?;

    let total = 24_000_000_000u64; // 24 GB
    let existing = vec![
        Reservation {
            tenant: "training".into(),
            bytes: 14_000_000_000,
        },
        Reservation {
            tenant: "eval".into(),
            bytes: 4_000_000_000,
        },
    ];

    for (label, new_bytes) in [
        ("happy 4GB", 4_000_000_000u64),
        ("too big 10GB", 10_000_000_000),
        ("violates margin 5.5GB", 5_500_000_000),
    ] {
        println!(
            "{label:>22}  →  {:?}",
            try_allocate(total, &existing, "newtenant", new_bytes)
        );
    }
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
    fn happy_allocation_passes() {
        let total = 24_000_000_000u64;
        let existing = vec![Reservation {
            tenant: "a".into(),
            bytes: 4_000_000_000,
        }];
        // 4GB used, 20GB free; allocate 10GB → 10GB free post-alloc (above 1GB margin).
        assert_eq!(
            try_allocate(total, &existing, "b", 10_000_000_000),
            AllocVerdict::Allocated
        );
    }

    #[test]
    fn allocation_exceeding_free_rejected() {
        let total = 24_000_000_000u64;
        let existing = vec![Reservation {
            tenant: "a".into(),
            bytes: 14_000_000_000,
        }];
        // 14 used, 10 free; need 15 → reject.
        let v = try_allocate(total, &existing, "b", 15_000_000_000);
        assert!(matches!(v, AllocVerdict::InsufficientFreeVram { .. }));
    }

    #[test]
    fn allocation_violating_safety_margin_rejected() {
        let total = 24_000_000_000u64;
        let existing = vec![Reservation {
            tenant: "a".into(),
            bytes: 14_000_000_000,
        }];
        // 14 used, 10 free; need 9.5 → 0.5 GB free post-alloc = below 1GB margin.
        assert_eq!(
            try_allocate(total, &existing, "b", 9_500_000_000),
            AllocVerdict::WouldViolateSafetyMargin
        );
    }

    #[test]
    fn empty_tenant_rejected() {
        let total = 24_000_000_000u64;
        assert_eq!(
            try_allocate(total, &[], "", 1_000_000_000),
            AllocVerdict::EmptyTenant
        );
    }

    #[test]
    fn no_existing_reservations_uses_full_vram() {
        let total = 24_000_000_000u64;
        // 23 GB allocation leaves 1 GB free = exactly at margin, passes.
        assert_eq!(
            try_allocate(total, &[], "a", 23_000_000_000),
            AllocVerdict::Allocated
        );
    }

    #[test]
    fn allocation_at_total_minus_margin_passes() {
        let total = 24_000_000_000u64;
        // 23 GB request → 1 GB free post-alloc = exactly at margin floor.
        assert_eq!(
            try_allocate(total, &[], "a", 23_000_000_000),
            AllocVerdict::Allocated
        );
    }

    #[test]
    fn over_total_returns_insufficient_not_negative() {
        let total = 1_000u64;
        let v = try_allocate(total, &[], "a", 999_999);
        assert!(matches!(v, AllocVerdict::InsufficientFreeVram { .. }));
    }
}
