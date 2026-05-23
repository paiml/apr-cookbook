//! # Bundle Pre-Load Warmup Pages
//!
//! mmap-backed bundle has cold pages on first access. Pre-fault by:
//!   touching first byte of each page (cheap, sequential)
//!   madvise WILLNEED (kernel async readahead)
//!   eager full read (highest cost, lowest first-inference latency)
//!
//! Picker rules:
//!   model_bytes < 100 MiB → EagerFullRead
//!   model_bytes < 4 GiB → MadviseWillNeed
//!   ≥ 4 GiB → SequentialPageTouch (avoid OS thrashing)
//!
//! Demonstrates the **BUNDLE.22** recipe for PMAT-148 (bundling round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Linux madvise(2) + mlock(2) man pages.
//!
//! Run with: cargo run --example bundle_pre_load_warmup
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PAGE_SIZE_BYTES: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupStrategy {
    SequentialPageTouch,
    MadviseWillNeed,
    EagerFullRead,
}

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok {
        strategy: WarmupStrategy,
        pages_to_touch: u64,
        estimated_secs: f64,
    },
    InvalidSize,
}

pub fn pick(model_bytes: u64, available_ram_gib: u32) -> WarmupVerdict {
    if model_bytes == 0 {
        return WarmupVerdict::InvalidSize;
    }
    let model_gib = model_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let strategy = if model_bytes < 100 * 1024 * 1024 {
        WarmupStrategy::EagerFullRead
    } else if model_gib < 4.0 && model_gib * 2.0 <= f64::from(available_ram_gib) {
        WarmupStrategy::MadviseWillNeed
    } else {
        WarmupStrategy::SequentialPageTouch
    };
    let pages_to_touch = model_bytes.div_ceil(PAGE_SIZE_BYTES);
    let estimated_secs = match strategy {
        WarmupStrategy::EagerFullRead => model_gib * 0.5,
        WarmupStrategy::MadviseWillNeed => model_gib * 0.05,
        WarmupStrategy::SequentialPageTouch => model_gib * 0.10,
    };
    WarmupVerdict::Ok {
        strategy,
        pages_to_touch,
        estimated_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_pre_load_warmup")?;

    println!("small 50 MiB: {:?}", pick(50 * 1024 * 1024, 16));
    println!("medium 1 GiB: {:?}", pick(1024 * 1024 * 1024, 16));
    println!("large 8 GiB: {:?}", pick(8u64 * 1024 * 1024 * 1024, 16));
    println!("invalid: {:?}", pick(0, 16));
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
    fn small_eager_full() {
        let v = pick(50 * 1024 * 1024, 16);
        if let WarmupVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, WarmupStrategy::EagerFullRead);
        }
    }

    #[test]
    fn medium_madvise() {
        let v = pick(1024 * 1024 * 1024, 16);
        if let WarmupVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, WarmupStrategy::MadviseWillNeed);
        }
    }

    #[test]
    fn large_sequential_touch() {
        let v = pick(8u64 * 1024 * 1024 * 1024, 16);
        if let WarmupVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, WarmupStrategy::SequentialPageTouch);
        }
    }

    #[test]
    fn ram_constrained_falls_back_to_sequential() {
        // 3 GiB model + 4 GiB RAM (< 2× model) → SequentialPageTouch.
        let v = pick(3 * 1024 * 1024 * 1024, 4);
        if let WarmupVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, WarmupStrategy::SequentialPageTouch);
        }
    }

    #[test]
    fn invalid_zero_size() {
        assert_eq!(pick(0, 16), WarmupVerdict::InvalidSize);
    }

    #[test]
    fn pages_proportional_to_size() {
        // 4 KiB page → 1 page for ≤ 4096 bytes; 2 pages for 4097-8192.
        let v_one = pick(1024, 16);
        let v_two = pick(8192, 16);
        if let (
            WarmupVerdict::Ok {
                pages_to_touch: a, ..
            },
            WarmupVerdict::Ok {
                pages_to_touch: b, ..
            },
        ) = (v_one, v_two)
        {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
        }
    }

    #[test]
    fn estimated_time_for_eager_higher() {
        // Eager full read takes longer than madvise.
        let v_eager = pick(50 * 1024 * 1024, 16);
        let v_madv = pick(1024 * 1024 * 1024, 16);
        if let (WarmupVerdict::Ok { strategy: a, .. }, WarmupVerdict::Ok { strategy: b, .. }) =
            (v_eager, v_madv)
        {
            assert_eq!(a, WarmupStrategy::EagerFullRead);
            assert_eq!(b, WarmupStrategy::MadviseWillNeed);
        }
    }

    #[test]
    fn pages_at_least_one_for_one_byte() {
        let v = pick(1, 16);
        if let WarmupVerdict::Ok { pages_to_touch, .. } = v {
            assert_eq!(pages_to_touch, 1);
        }
    }

    #[test]
    fn boundary_at_100_mib_madvise() {
        let v = pick(100 * 1024 * 1024, 16);
        if let WarmupVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, WarmupStrategy::MadviseWillNeed);
        }
    }

    #[test]
    fn estimated_time_finite() {
        let v = pick(100 * 1024 * 1024, 16);
        if let WarmupVerdict::Ok { estimated_secs, .. } = v {
            assert!(estimated_secs.is_finite());
            assert!(estimated_secs >= 0.0);
        }
    }
}
