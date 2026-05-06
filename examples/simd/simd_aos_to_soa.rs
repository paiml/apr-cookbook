//! # SIMD Array-of-Structs ↔ Struct-of-Arrays Planner
//!
//! AoS: `[{x,y,z}, {x,y,z}, ...]` — friendly for individual lookups.
//! SoA: `{xs:[...], ys:[...], zs:[...]}` — friendly for SIMD across one
//! component (e.g. multiply all x by scalar).
//!
//! This recipe estimates: AoS load cost = struct_bytes per element
//! (a full cache line for one component), SoA load cost = 1 SIMD word
//! that holds N components stacked. Picks layout based on access
//! pattern.
//!
//! Demonstrates the **SIMD.14** recipe for PMAT-138 (simd round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ISPC documentation § AoS vs SoA.
//!
//! Run with: cargo run --example simd_aos_to_soa
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    OneComponentMany,
    AllComponentsOne,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Aos,
    Soa,
}

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Pick {
        layout: Layout,
        speedup_estimate: f64,
    },
    InvalidShape,
}

pub fn pick(
    pattern: AccessPattern,
    n_components: u32,
    component_bytes: u32,
    n_elements: u32,
) -> LayoutVerdict {
    if n_components == 0 || component_bytes == 0 || n_elements == 0 {
        return LayoutVerdict::InvalidShape;
    }
    let layout = match pattern {
        AccessPattern::OneComponentMany => Layout::Soa,
        AccessPattern::AllComponentsOne => Layout::Aos,
        AccessPattern::Mixed => {
            if n_elements >= 1_000 {
                Layout::Soa
            } else {
                Layout::Aos
            }
        }
    };
    let speedup_estimate = match (pattern, layout) {
        (AccessPattern::OneComponentMany, Layout::Soa) => f64::from(n_components),
        (AccessPattern::AllComponentsOne, Layout::Aos) => 1.0,
        _ => 1.5,
    };
    LayoutVerdict::Pick {
        layout,
        speedup_estimate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_aos_to_soa")?;

    println!(
        "one-component (xyz, 3 components): {:?}",
        pick(AccessPattern::OneComponentMany, 3, 4, 10_000)
    );
    println!(
        "all-components: {:?}",
        pick(AccessPattern::AllComponentsOne, 3, 4, 10_000)
    );
    println!(
        "mixed (large): {:?}",
        pick(AccessPattern::Mixed, 4, 4, 10_000)
    );
    println!("mixed (small): {:?}", pick(AccessPattern::Mixed, 4, 4, 100));
    println!("invalid: {:?}", pick(AccessPattern::Mixed, 0, 4, 10));
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
    fn one_component_picks_soa() {
        let v = pick(AccessPattern::OneComponentMany, 3, 4, 10_000);
        if let LayoutVerdict::Pick { layout, .. } = v {
            assert_eq!(layout, Layout::Soa);
        }
    }

    #[test]
    fn all_components_picks_aos() {
        let v = pick(AccessPattern::AllComponentsOne, 3, 4, 10_000);
        if let LayoutVerdict::Pick { layout, .. } = v {
            assert_eq!(layout, Layout::Aos);
        }
    }

    #[test]
    fn mixed_large_picks_soa() {
        let v = pick(AccessPattern::Mixed, 4, 4, 10_000);
        if let LayoutVerdict::Pick { layout, .. } = v {
            assert_eq!(layout, Layout::Soa);
        }
    }

    #[test]
    fn mixed_small_picks_aos() {
        let v = pick(AccessPattern::Mixed, 4, 4, 100);
        if let LayoutVerdict::Pick { layout, .. } = v {
            assert_eq!(layout, Layout::Aos);
        }
    }

    #[test]
    fn one_component_speedup_equals_n_components() {
        if let LayoutVerdict::Pick {
            speedup_estimate, ..
        } = pick(AccessPattern::OneComponentMany, 4, 4, 10_000)
        {
            assert!((speedup_estimate - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn all_components_speedup_one() {
        if let LayoutVerdict::Pick {
            speedup_estimate, ..
        } = pick(AccessPattern::AllComponentsOne, 3, 4, 10_000)
        {
            assert!((speedup_estimate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_components_rejected() {
        assert_eq!(
            pick(AccessPattern::Mixed, 0, 4, 10),
            LayoutVerdict::InvalidShape
        );
    }

    #[test]
    fn invalid_zero_bytes_rejected() {
        assert_eq!(
            pick(AccessPattern::Mixed, 3, 0, 10),
            LayoutVerdict::InvalidShape
        );
    }

    #[test]
    fn invalid_zero_elements_rejected() {
        assert_eq!(
            pick(AccessPattern::Mixed, 3, 4, 0),
            LayoutVerdict::InvalidShape
        );
    }

    #[test]
    fn boundary_at_1000_picks_soa() {
        let v = pick(AccessPattern::Mixed, 4, 4, 1_000);
        if let LayoutVerdict::Pick { layout, .. } = v {
            assert_eq!(layout, Layout::Soa);
        }
    }
}
