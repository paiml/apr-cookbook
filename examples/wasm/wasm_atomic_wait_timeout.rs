//! # WASM Atomic Wait Timeout
//!
//! Validate `memory.atomic.wait32` parameters: timeout in nanoseconds
//! ≥ 0 (unsigned), addr aligned to 4 bytes. Returns categorical
//! verdict.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly threads proposal §4.5 atomic.wait/notify;
//!  pthread_cond_timedwait equivalence.
//!
//! Run with: cargo run --example wasm_atomic_wait_timeout
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AtomicWaitVerdict {
    Valid,
    UnalignedAddr,
    InvalidTimeout,
    InvalidConfig,
}

pub fn validate(addr: u32, timeout_ns: i64) -> AtomicWaitVerdict {
    if addr % 4 != 0 {
        return AtomicWaitVerdict::UnalignedAddr;
    }
    if timeout_ns < -1 {
        return AtomicWaitVerdict::InvalidTimeout;
    }
    AtomicWaitVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_atomic_wait_timeout")?;

    println!("valid: {:?}", validate(0, 1_000_000));
    println!("indefinite (-1): {:?}", validate(0, -1));
    println!("unaligned: {:?}", validate(3, 1_000_000));
    println!("invalid timeout: {:?}", validate(0, -2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn aligned_addr_with_positive_timeout_valid() {
        assert_eq!(validate(0, 1_000_000), AtomicWaitVerdict::Valid);
    }

    #[test]
    fn aligned_addr_with_indefinite_valid() {
        assert_eq!(validate(0, -1), AtomicWaitVerdict::Valid);
    }

    #[test]
    fn unaligned_addr_rejected() {
        assert_eq!(validate(3, 1000), AtomicWaitVerdict::UnalignedAddr);
    }

    #[test]
    fn timeout_below_minus_one_rejected() {
        assert_eq!(validate(0, -2), AtomicWaitVerdict::InvalidTimeout);
    }

    #[test]
    fn deterministic() {
        let r1 = validate(4, 100);
        let r2 = validate(4, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn aligned_4_8_12_valid() {
        for addr in [4u32, 8, 12, 16, 100] {
            assert_eq!(validate(addr, 100), AtomicWaitVerdict::Valid);
        }
    }

    #[test]
    fn unaligned_1_2_3_5_rejected() {
        for addr in [1u32, 2, 3, 5] {
            assert_eq!(validate(addr, 100), AtomicWaitVerdict::UnalignedAddr);
        }
    }

    #[test]
    fn zero_timeout_valid() {
        assert_eq!(validate(0, 0), AtomicWaitVerdict::Valid);
    }

    #[test]
    fn high_timeout_valid() {
        assert_eq!(validate(0, i64::MAX), AtomicWaitVerdict::Valid);
    }

    #[test]
    fn very_negative_timeout_invalid() {
        assert_eq!(validate(0, i64::MIN), AtomicWaitVerdict::InvalidTimeout);
    }

    #[test]
    fn high_addr_aligned_valid() {
        assert_eq!(validate(0xFFFF_FFFC, 0), AtomicWaitVerdict::Valid);
    }

    #[test]
    fn boundary_addr_zero_aligned() {
        assert_eq!(validate(0, 0), AtomicWaitVerdict::Valid);
    }
}
