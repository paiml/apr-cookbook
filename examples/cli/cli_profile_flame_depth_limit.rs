//! # apr profile --flame-graph-depth — Stack Depth Cap
//!
//! Flame graphs become unreadable beyond ~50 frames; collection cost
//! also scales linearly with depth. `apr profile --flame-graph-depth <N>`
//! caps unwinding. Floor: 10 (too shallow misses hot paths); default:
//! 64 (deep enough for most async stacks); ceiling: 256 (further hits
//! libunwind cost cliff).
//!
//! Demonstrates the **PROF.5** recipe for PMAT-115 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROF-001 + Gregg 2016 + DTrace ustack semantics
//!
//! Run with: cargo run --example cli_profile_flame_depth_limit
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DepthVerdict {
    Ok,
    TooShallow { recommended: u32 },
    TooDeep { recommended: u32 },
    InvalidZero,
}

const MIN_DEPTH: u32 = 10;
const DEFAULT_DEPTH: u32 = 64;
const MAX_DEPTH: u32 = 256;

pub fn classify(depth: u32) -> DepthVerdict {
    if depth == 0 {
        return DepthVerdict::InvalidZero;
    }
    if depth < MIN_DEPTH {
        return DepthVerdict::TooShallow {
            recommended: DEFAULT_DEPTH,
        };
    }
    if depth > MAX_DEPTH {
        return DepthVerdict::TooDeep {
            recommended: MAX_DEPTH,
        };
    }
    DepthVerdict::Ok
}

pub fn was_truncated(observed_depth: u32, max_depth: u32) -> bool {
    observed_depth >= max_depth
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_flame_depth_limit")?;

    for d in [0u32, 5, 10, 64, 256, 512] {
        println!("depth={d:>4}  →  {:?}", classify(d));
    }
    println!("truncated(observed=64, max=64)? {}", was_truncated(64, 64));
    println!("truncated(observed=63, max=64)? {}", was_truncated(63, 64));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(classify(0), DepthVerdict::InvalidZero);
    }

    #[test]
    fn under_floor_rejected() {
        let v = classify(5);
        assert!(matches!(v, DepthVerdict::TooShallow { .. }));
    }

    #[test]
    fn at_floor_passes() {
        assert_eq!(classify(MIN_DEPTH), DepthVerdict::Ok);
    }

    #[test]
    fn default_passes() {
        assert_eq!(classify(DEFAULT_DEPTH), DepthVerdict::Ok);
    }

    #[test]
    fn at_ceiling_passes() {
        assert_eq!(classify(MAX_DEPTH), DepthVerdict::Ok);
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = classify(512);
        assert!(matches!(v, DepthVerdict::TooDeep { .. }));
    }

    #[test]
    fn truncation_at_or_above_limit() {
        // Equality means at-limit, signaling possible truncation.
        assert!(was_truncated(64, 64));
        assert!(was_truncated(100, 64));
    }

    #[test]
    fn no_truncation_below_limit() {
        assert!(!was_truncated(63, 64));
        assert!(!was_truncated(0, 64));
    }
}
