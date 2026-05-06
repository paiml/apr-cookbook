//! # Registry Version-Constraint Resolver
//!
//! Pip/Cargo-style constraints:
//! - eq 1.2.3 → exact
//! - caret 1.2.3 → 1.x.x compatible (within major)
//! - ge 1.2.3 → at least
//! - lt 2.0.0 → less than
//!
//! Resolver: given a constraint and available versions, returns the
//! best-match version (highest matching).
//!
//! Demonstrates the **REG.20** recipe for PMAT-147 (registry round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pip PEP 440 + Cargo semver semantics.
//!
//! Run with: cargo run --example registry_dependency_resolver
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Exact,
    Caret,
    GreaterEqual,
    LessThan,
}

#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub bound: Version,
}

#[derive(Debug, PartialEq)]
pub enum ResolveVerdict {
    Resolved(Version),
    NoMatch,
    EmptyAvailable,
}

pub fn resolve(constraint: Constraint, available: &[Version]) -> ResolveVerdict {
    if available.is_empty() {
        return ResolveVerdict::EmptyAvailable;
    }
    let mut best: Option<Version> = None;
    for v in available {
        if matches_constraint(constraint, *v) {
            best = match best {
                None => Some(*v),
                Some(b) => Some(if *v > b { *v } else { b }),
            };
        }
    }
    match best {
        Some(v) => ResolveVerdict::Resolved(v),
        None => ResolveVerdict::NoMatch,
    }
}

fn matches_constraint(c: Constraint, v: Version) -> bool {
    match c.kind {
        ConstraintKind::Exact => v == c.bound,
        ConstraintKind::Caret => {
            v >= c.bound
                && v.major == c.bound.major
                && (c.bound.major != 0 || v.minor == c.bound.minor)
        }
        ConstraintKind::GreaterEqual => v >= c.bound,
        ConstraintKind::LessThan => v < c.bound,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_dependency_resolver")?;

    let versions = [
        Version {
            major: 1,
            minor: 0,
            patch: 0,
        },
        Version {
            major: 1,
            minor: 2,
            patch: 3,
        },
        Version {
            major: 1,
            minor: 5,
            patch: 0,
        },
        Version {
            major: 2,
            minor: 0,
            patch: 0,
        },
    ];

    println!(
        "^1.2.3: {:?}",
        resolve(
            Constraint {
                kind: ConstraintKind::Caret,
                bound: Version {
                    major: 1,
                    minor: 2,
                    patch: 3
                }
            },
            &versions
        )
    );
    println!(
        ">= 1.5.0: {:?}",
        resolve(
            Constraint {
                kind: ConstraintKind::GreaterEqual,
                bound: Version {
                    major: 1,
                    minor: 5,
                    patch: 0
                }
            },
            &versions
        )
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions() -> Vec<Version> {
        vec![
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            Version {
                major: 1,
                minor: 2,
                patch: 3,
            },
            Version {
                major: 1,
                minor: 5,
                patch: 0,
            },
            Version {
                major: 2,
                minor: 0,
                patch: 0,
            },
        ]
    }

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match() {
        let c = Constraint {
            kind: ConstraintKind::Exact,
            bound: Version {
                major: 1,
                minor: 2,
                patch: 3,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            assert_eq!(v.minor, 2);
            assert_eq!(v.patch, 3);
        }
    }

    #[test]
    fn caret_picks_highest_in_major() {
        // ^1.2.3 → 1.x.x, picks 1.5.0.
        let c = Constraint {
            kind: ConstraintKind::Caret,
            bound: Version {
                major: 1,
                minor: 2,
                patch: 3,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            assert_eq!(v.minor, 5);
        }
    }

    #[test]
    fn caret_excludes_next_major() {
        let c = Constraint {
            kind: ConstraintKind::Caret,
            bound: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            // Should not pick 2.0.0.
            assert!(v.major < 2);
        }
    }

    #[test]
    fn greater_equal_picks_highest() {
        let c = Constraint {
            kind: ConstraintKind::GreaterEqual,
            bound: Version {
                major: 1,
                minor: 5,
                patch: 0,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            assert_eq!(v.major, 2);
        }
    }

    #[test]
    fn less_than_picks_highest_below() {
        let c = Constraint {
            kind: ConstraintKind::LessThan,
            bound: Version {
                major: 2,
                minor: 0,
                patch: 0,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            assert_eq!(v.minor, 5);
        }
    }

    #[test]
    fn no_match_returns_no_match() {
        let c = Constraint {
            kind: ConstraintKind::Exact,
            bound: Version {
                major: 9,
                minor: 9,
                patch: 9,
            },
        };
        assert_eq!(resolve(c, &versions()), ResolveVerdict::NoMatch);
    }

    #[test]
    fn empty_available_rejected() {
        let c = Constraint {
            kind: ConstraintKind::Caret,
            bound: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
        };
        assert_eq!(resolve(c, &[]), ResolveVerdict::EmptyAvailable);
    }

    #[test]
    fn caret_zero_major_pins_minor() {
        // ^0.2.3 means >= 0.2.3, < 0.3.0 (zero-major has tighter rule).
        let v_022 = Version {
            major: 0,
            minor: 2,
            patch: 2,
        };
        let v_023 = Version {
            major: 0,
            minor: 2,
            patch: 3,
        };
        let v_030 = Version {
            major: 0,
            minor: 3,
            patch: 0,
        };
        let c = Constraint {
            kind: ConstraintKind::Caret,
            bound: v_023,
        };
        let avail = vec![v_022, v_023, v_030];
        if let ResolveVerdict::Resolved(v) = resolve(c, &avail) {
            // Should NOT pick 0.3.0.
            assert_eq!(v, v_023);
        }
    }

    #[test]
    fn version_ord_correct() {
        let a = Version {
            major: 1,
            minor: 2,
            patch: 3,
        };
        let b = Version {
            major: 1,
            minor: 2,
            patch: 4,
        };
        assert!(a < b);
    }

    #[test]
    fn caret_excludes_below_bound() {
        // ^1.2.3 should not include 1.0.0.
        let c = Constraint {
            kind: ConstraintKind::Caret,
            bound: Version {
                major: 1,
                minor: 2,
                patch: 3,
            },
        };
        if let ResolveVerdict::Resolved(v) = resolve(c, &versions()) {
            assert!(v.minor >= 2);
        }
    }
}
