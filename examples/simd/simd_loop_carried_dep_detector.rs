//! # SIMD Loop-Carried Dependency Detector
//!
//! Patterns that block auto-vectorization:
//!   ReadAfterWrite (RAW): a[i] = a[i-1] + x  → distance 1 read-after-write
//!   WriteAfterWrite (WAW): a[i] = x; a[i] = y  → not actually a problem (last wins)
//!   WriteAfterRead (WAR): t = a[i]; a[i+1] = t  → forward write OK
//!   IndirectIndex: a[idx[i]]  → cannot prove distance, blocks vectorization
//!
//! This recipe classifies a pattern + recommends a remedy.
//!
//! Demonstrates the **SIMD.12** recipe for PMAT-138 (simd round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Allen-Kennedy. Optimizing Compilers for Modern Architectures.
//!
//! Run with: cargo run --example simd_loop_carried_dep_detector
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    ReadAfterWrite,
    WriteAfterWrite,
    WriteAfterRead,
    IndirectIndex,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Remedy {
    Vectorize,
    Unroll,
    Scalarize,
    GatherScatter,
}

#[derive(Debug, PartialEq)]
pub enum DepVerdict {
    Vectorizable { kind: DepKind, remedy: Remedy },
    Blocked { kind: DepKind, remedy: Remedy },
    InvalidPattern,
}

pub fn classify(kind: DepKind, distance: i32) -> DepVerdict {
    match kind {
        DepKind::None => DepVerdict::Vectorizable {
            kind: DepKind::None,
            remedy: Remedy::Vectorize,
        },
        DepKind::ReadAfterWrite => {
            if distance.abs() >= 16 {
                DepVerdict::Vectorizable {
                    kind: DepKind::ReadAfterWrite,
                    remedy: Remedy::Vectorize,
                }
            } else if distance.abs() >= 4 {
                DepVerdict::Vectorizable {
                    kind: DepKind::ReadAfterWrite,
                    remedy: Remedy::Unroll,
                }
            } else {
                DepVerdict::Blocked {
                    kind: DepKind::ReadAfterWrite,
                    remedy: Remedy::Scalarize,
                }
            }
        }
        DepKind::WriteAfterWrite => DepVerdict::Vectorizable {
            kind: DepKind::WriteAfterWrite,
            remedy: Remedy::Vectorize,
        },
        DepKind::WriteAfterRead => {
            if distance >= 0 {
                DepVerdict::Vectorizable {
                    kind: DepKind::WriteAfterRead,
                    remedy: Remedy::Vectorize,
                }
            } else {
                DepVerdict::Blocked {
                    kind: DepKind::WriteAfterRead,
                    remedy: Remedy::Scalarize,
                }
            }
        }
        DepKind::IndirectIndex => DepVerdict::Blocked {
            kind: DepKind::IndirectIndex,
            remedy: Remedy::GatherScatter,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_loop_carried_dep_detector")?;

    let cases = [
        (DepKind::None, 0i32),
        (DepKind::ReadAfterWrite, 1),
        (DepKind::ReadAfterWrite, 8),
        (DepKind::ReadAfterWrite, 32),
        (DepKind::WriteAfterWrite, 0),
        (DepKind::WriteAfterRead, 1),
        (DepKind::WriteAfterRead, -1),
        (DepKind::IndirectIndex, 0),
    ];
    for (k, d) in cases {
        println!("{k:?} dist={d} → {:?}", classify(k, d));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_dep_vectorizes_directly() {
        let v = classify(DepKind::None, 0);
        assert_eq!(
            v,
            DepVerdict::Vectorizable {
                kind: DepKind::None,
                remedy: Remedy::Vectorize
            }
        );
    }

    #[test]
    fn raw_short_distance_blocked() {
        let v = classify(DepKind::ReadAfterWrite, 1);
        assert!(matches!(
            v,
            DepVerdict::Blocked {
                kind: DepKind::ReadAfterWrite,
                ..
            }
        ));
    }

    #[test]
    fn raw_medium_distance_unrolls() {
        let v = classify(DepKind::ReadAfterWrite, 8);
        assert!(matches!(
            v,
            DepVerdict::Vectorizable {
                kind: DepKind::ReadAfterWrite,
                remedy: Remedy::Unroll
            }
        ));
    }

    #[test]
    fn raw_long_distance_full_vectorize() {
        let v = classify(DepKind::ReadAfterWrite, 32);
        assert!(matches!(
            v,
            DepVerdict::Vectorizable {
                kind: DepKind::ReadAfterWrite,
                remedy: Remedy::Vectorize
            }
        ));
    }

    #[test]
    fn waw_always_vectorizes() {
        let v = classify(DepKind::WriteAfterWrite, 0);
        assert!(matches!(v, DepVerdict::Vectorizable { .. }));
    }

    #[test]
    fn war_forward_distance_vectorizes() {
        let v = classify(DepKind::WriteAfterRead, 1);
        assert!(matches!(v, DepVerdict::Vectorizable { .. }));
    }

    #[test]
    fn war_backward_distance_blocked() {
        let v = classify(DepKind::WriteAfterRead, -1);
        assert!(matches!(v, DepVerdict::Blocked { .. }));
    }

    #[test]
    fn indirect_index_uses_gather_scatter() {
        let v = classify(DepKind::IndirectIndex, 0);
        assert_eq!(
            v,
            DepVerdict::Blocked {
                kind: DepKind::IndirectIndex,
                remedy: Remedy::GatherScatter
            }
        );
    }

    #[test]
    fn raw_negative_distance_uses_abs() {
        // |−1| = 1, still blocked.
        let v = classify(DepKind::ReadAfterWrite, -1);
        assert!(matches!(v, DepVerdict::Blocked { .. }));
    }

    #[test]
    fn raw_at_unroll_threshold_unrolls() {
        // distance = 4 → exactly the unroll threshold.
        let v = classify(DepKind::ReadAfterWrite, 4);
        assert!(matches!(
            v,
            DepVerdict::Vectorizable {
                remedy: Remedy::Unroll,
                ..
            }
        ));
    }

    #[test]
    fn raw_at_vectorize_threshold_vectorizes() {
        let v = classify(DepKind::ReadAfterWrite, 16);
        assert!(matches!(
            v,
            DepVerdict::Vectorizable {
                remedy: Remedy::Vectorize,
                ..
            }
        ));
    }
}
