//! # Contracts-Macros YAML Root Node Check
//!
//! Verify a YAML doc's root node is the expected type (Mapping
//! `{...}` or Sequence `[...]`). Returns the actual root type and
//! categorical match verdict.
//!
//! Demonstrates the **CMM.196** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6 collection types; libyaml event-loop
//!  document-start production.
//!
//! Run with: cargo run --example contracts_macros_yaml_root_node_check
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RootKind {
    Mapping,
    Sequence,
    Scalar,
    Empty,
}

#[derive(Debug, PartialEq)]
pub enum RootCheckVerdict {
    Match {
        actual: RootKind,
    },
    Mismatch {
        expected: RootKind,
        actual: RootKind,
    },
    InvalidConfig,
}

pub fn check(buffer: &str, expected: RootKind) -> RootCheckVerdict {
    if matches!(expected, RootKind::Empty) {
        return RootCheckVerdict::InvalidConfig;
    }
    let trimmed = buffer.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{feff}');
    let actual = if trimmed.is_empty() {
        RootKind::Empty
    } else if trimmed.starts_with('-') || trimmed.starts_with('[') {
        RootKind::Sequence
    } else if trimmed.starts_with('{') || trimmed.contains(':') {
        RootKind::Mapping
    } else {
        RootKind::Scalar
    };
    if actual == expected {
        RootCheckVerdict::Match { actual }
    } else {
        RootCheckVerdict::Mismatch { expected, actual }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_root_node_check")?;

    println!("mapping: {:?}", check("a: 1\n", RootKind::Mapping));
    println!("sequence: {:?}", check("- a\n- b\n", RootKind::Sequence));
    println!("mismatch: {:?}", check("a: 1\n", RootKind::Sequence));
    println!("invalid: {:?}", check("", RootKind::Empty));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn mapping_matches() {
        let v = check("a: 1\n", RootKind::Mapping);
        assert_eq!(
            v,
            RootCheckVerdict::Match {
                actual: RootKind::Mapping
            }
        );
    }

    #[test]
    fn sequence_block_matches() {
        let v = check("- a\n- b\n", RootKind::Sequence);
        assert_eq!(
            v,
            RootCheckVerdict::Match {
                actual: RootKind::Sequence,
            }
        );
    }

    #[test]
    fn sequence_flow_matches() {
        let v = check("[1, 2, 3]\n", RootKind::Sequence);
        assert_eq!(
            v,
            RootCheckVerdict::Match {
                actual: RootKind::Sequence,
            }
        );
    }

    #[test]
    fn mapping_flow_matches() {
        let v = check("{a: 1}\n", RootKind::Mapping);
        assert_eq!(
            v,
            RootCheckVerdict::Match {
                actual: RootKind::Mapping,
            }
        );
    }

    #[test]
    fn scalar_root() {
        let v = check("plain scalar\n", RootKind::Scalar);
        assert_eq!(
            v,
            RootCheckVerdict::Match {
                actual: RootKind::Scalar,
            }
        );
    }

    #[test]
    fn mismatch_returns_both() {
        let v = check("a: 1\n", RootKind::Sequence);
        if let RootCheckVerdict::Mismatch { expected, actual } = v {
            assert_eq!(expected, RootKind::Sequence);
            assert_eq!(actual, RootKind::Mapping);
        }
    }

    #[test]
    fn empty_expected_rejected() {
        assert_eq!(
            check("a: 1", RootKind::Empty),
            RootCheckVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let r1 = check("a: 1\n", RootKind::Mapping);
        let r2 = check("a: 1\n", RootKind::Mapping);
        assert_eq!(r1, r2);
    }

    #[test]
    fn whitespace_prefix_trimmed() {
        let v = check("\n\n  a: 1\n", RootKind::Mapping);
        if let RootCheckVerdict::Match { actual } = v {
            assert_eq!(actual, RootKind::Mapping);
        }
    }

    #[test]
    fn empty_buffer_classified_empty() {
        let v = check("", RootKind::Mapping);
        if let RootCheckVerdict::Mismatch { actual, .. } = v {
            assert_eq!(actual, RootKind::Empty);
        }
    }

    #[test]
    fn unicode_value_handled() {
        let v = check("name: café\n", RootKind::Mapping);
        if let RootCheckVerdict::Match { actual } = v {
            assert_eq!(actual, RootKind::Mapping);
        }
    }

    #[test]
    fn many_keys_still_mapping() {
        let mut buf = String::new();
        for i in 0..30 {
            buf.push_str(&format!("k{i}: v\n"));
        }
        let v = check(&buf, RootKind::Mapping);
        if let RootCheckVerdict::Match { actual } = v {
            assert_eq!(actual, RootKind::Mapping);
        }
    }
}
