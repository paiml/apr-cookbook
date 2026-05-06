//! # Acceleration Kernel Dispatch Planner
//!
//! Pick the fastest kernel variant for a given problem size + ISA
//! capabilities. Heuristic: scalar for tiny (< 64 elems, setup cost
//! dominates); SSE for small (64-1023); AVX2 for medium; AVX-512 for
//! large + AVX-512 available. This recipe builds the planner.
//!
//! Demonstrates the **ACCEL.4** recipe for PMAT-126 (acceleration coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Williams et al. (2009). Roofline: an insightful visual performance model.
//!
//! Run with: cargo run --example acceleration_kernel_dispatch_planner
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct IsaSet {
    pub sse: bool,
    pub avx2: bool,
    pub avx512: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelChoice {
    Scalar,
    Sse,
    Avx2,
    Avx512,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok(KernelChoice),
    InvalidSize,
}

const SCALAR_CEILING: usize = 63;
const SSE_CEILING: usize = 1023;
const AVX2_CEILING: usize = 65_535;

pub fn dispatch(num_elements: usize, isa: IsaSet) -> DispatchVerdict {
    if num_elements == 0 {
        return DispatchVerdict::InvalidSize;
    }
    let pick = if num_elements <= SCALAR_CEILING {
        KernelChoice::Scalar
    } else if num_elements <= SSE_CEILING && isa.sse {
        KernelChoice::Sse
    } else if num_elements <= AVX2_CEILING && isa.avx2 {
        KernelChoice::Avx2
    } else if isa.avx512 {
        KernelChoice::Avx512
    } else if isa.avx2 {
        KernelChoice::Avx2
    } else if isa.sse {
        KernelChoice::Sse
    } else {
        KernelChoice::Scalar
    };
    DispatchVerdict::Ok(pick)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("acceleration_kernel_dispatch_planner")?;

    let modern = IsaSet {
        sse: true,
        avx2: true,
        avx512: true,
    };
    let legacy = IsaSet {
        sse: true,
        ..Default::default()
    };
    for n in [16usize, 100, 5000, 100_000, 0] {
        println!("n={n} modern: {:?}", dispatch(n, modern));
        println!("n={n} legacy: {:?}", dispatch(n, legacy));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modern() -> IsaSet {
        IsaSet {
            sse: true,
            avx2: true,
            avx512: true,
        }
    }

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn tiny_uses_scalar() {
        // < 64 elems → setup cost dominates.
        assert_eq!(
            dispatch(16, modern()),
            DispatchVerdict::Ok(KernelChoice::Scalar)
        );
    }

    #[test]
    fn small_uses_sse() {
        // 100 elems → SSE wins for small workloads.
        assert_eq!(
            dispatch(100, modern()),
            DispatchVerdict::Ok(KernelChoice::Sse)
        );
    }

    #[test]
    fn medium_uses_avx2() {
        assert_eq!(
            dispatch(5000, modern()),
            DispatchVerdict::Ok(KernelChoice::Avx2)
        );
    }

    #[test]
    fn large_uses_avx512_when_available() {
        assert_eq!(
            dispatch(100_000, modern()),
            DispatchVerdict::Ok(KernelChoice::Avx512)
        );
    }

    #[test]
    fn large_falls_back_to_avx2_without_avx512() {
        let isa = IsaSet {
            sse: true,
            avx2: true,
            avx512: false,
        };
        assert_eq!(
            dispatch(100_000, isa),
            DispatchVerdict::Ok(KernelChoice::Avx2)
        );
    }

    #[test]
    fn falls_back_to_scalar_with_no_isa() {
        let isa = IsaSet::default();
        assert_eq!(
            dispatch(100_000, isa),
            DispatchVerdict::Ok(KernelChoice::Scalar)
        );
    }

    #[test]
    fn small_falls_back_to_scalar_when_sse_missing() {
        // Want SSE for 100 elems but ISA says no — no AVX2 either, fall to scalar.
        let isa = IsaSet::default();
        assert_eq!(
            dispatch(100, isa),
            DispatchVerdict::Ok(KernelChoice::Scalar)
        );
    }

    #[test]
    fn zero_elements_invalid() {
        assert_eq!(dispatch(0, modern()), DispatchVerdict::InvalidSize);
    }

    #[test]
    fn boundary_at_scalar_ceiling() {
        assert_eq!(
            dispatch(SCALAR_CEILING, modern()),
            DispatchVerdict::Ok(KernelChoice::Scalar)
        );
        assert_eq!(
            dispatch(SCALAR_CEILING + 1, modern()),
            DispatchVerdict::Ok(KernelChoice::Sse)
        );
    }
}
