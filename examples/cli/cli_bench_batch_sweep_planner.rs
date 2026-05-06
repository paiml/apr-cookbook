//! # apr bench --batch-sweep — Batch Size Sweep Planner
//!
//! `apr bench --batch-sweep <START..END>` walks batch sizes to find
//! the throughput knee. Strategy: powers-of-2 (1, 2, 4, 8, ...) up to
//! min(end, 256). Constraints: start ≥ 1; end ≤ 4096; start ≤ end.
//! This recipe builds the planner.
//!
//! Demonstrates the **BENCH.5** recipe for PMAT-118 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender BENCH-001 + roofline model literature
//!
//! Run with: cargo run --example cli_bench_batch_sweep_planner
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SweepVerdict {
    Ok(Vec<u32>),
    StartZero,
    EndExceedsCap { end: u32, cap: u32 },
    StartAfterEnd { start: u32, end: u32 },
}

const HARD_CAP: u32 = 4096;

pub fn plan(start: u32, end: u32) -> SweepVerdict {
    if start == 0 {
        return SweepVerdict::StartZero;
    }
    if end > HARD_CAP {
        return SweepVerdict::EndExceedsCap { end, cap: HARD_CAP };
    }
    if start > end {
        return SweepVerdict::StartAfterEnd { start, end };
    }
    let mut sweep = Vec::new();
    // First snap to a power of 2 that's ≥ start.
    let mut p: u32 = 1;
    while p < start {
        let Some(n) = p.checked_mul(2) else {
            return SweepVerdict::EndExceedsCap { end, cap: HARD_CAP };
        };
        p = n;
    }
    while p <= end {
        sweep.push(p);
        let Some(next) = p.checked_mul(2) else {
            break;
        };
        p = next;
    }
    SweepVerdict::Ok(sweep)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_batch_sweep_planner")?;

    let cases = [(1u32, 32), (4, 64), (3, 16), (0, 8), (1, 5000), (10, 5)];
    for (s, e) in cases {
        println!("[{s}..={e}]  →  {:?}", plan(s, e));
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
    fn typical_sweep_doubles() {
        let v = plan(1, 32);
        assert_eq!(v, SweepVerdict::Ok(vec![1, 2, 4, 8, 16, 32]));
    }

    #[test]
    fn start_snaps_to_power_of_two() {
        // start=3 → first sweep step is 4.
        if let SweepVerdict::Ok(s) = plan(3, 16) {
            assert_eq!(s[0], 4);
        }
    }

    #[test]
    fn end_clamps_to_largest_power_under_cap() {
        // 4..64 → [4, 8, 16, 32, 64].
        let v = plan(4, 64);
        assert_eq!(v, SweepVerdict::Ok(vec![4, 8, 16, 32, 64]));
    }

    #[test]
    fn start_zero_rejected() {
        assert_eq!(plan(0, 32), SweepVerdict::StartZero);
    }

    #[test]
    fn end_exceeds_cap_rejected() {
        let v = plan(1, 5000);
        assert!(matches!(v, SweepVerdict::EndExceedsCap { .. }));
    }

    #[test]
    fn start_after_end_rejected() {
        let v = plan(10, 5);
        assert!(matches!(v, SweepVerdict::StartAfterEnd { .. }));
    }

    #[test]
    fn at_cap_passes() {
        // 1024..4096 → [1024, 2048, 4096].
        let v = plan(1024, 4096);
        assert_eq!(v, SweepVerdict::Ok(vec![1024, 2048, 4096]));
    }

    #[test]
    fn start_equals_end_yields_one_step() {
        let v = plan(8, 8);
        assert_eq!(v, SweepVerdict::Ok(vec![8]));
    }

    #[test]
    fn all_results_are_powers_of_two() {
        if let SweepVerdict::Ok(s) = plan(1, 256) {
            for b in s {
                assert!(b.is_power_of_two());
            }
        }
    }
}
