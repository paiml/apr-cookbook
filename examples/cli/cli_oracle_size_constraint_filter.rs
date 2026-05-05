//! # apr oracle — Size Constraint Filter
//!
//! `apr oracle --family qwen2 --size 7b` filters the family contract to
//! the size variant that actually applies. Without `--size`, the oracle
//! reports the family-wide schema (any size). With it, the oracle locks
//! exact (n_layers, hidden, n_kv_heads) values for the variant. This
//! recipe builds the variant table and exposes the lookup function so a
//! CI pipeline can preview which constraints will apply.
//!
//! Demonstrates the **ORACLE.5** recipe for PMAT-093 (apr oracle coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ORACLE-003 + Hugging Face model-card spec
//!
//! Run with: cargo run --example cli_oracle_size_constraint_filter
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeVariant {
    pub family: &'static str,
    pub size: &'static str,
    pub n_layers: u32,
    pub hidden: u32,
    pub n_kv_heads: u32,
}

const VARIANTS: &[SizeVariant] = &[
    SizeVariant {
        family: "qwen2",
        size: "0.5b",
        n_layers: 24,
        hidden: 896,
        n_kv_heads: 2,
    },
    SizeVariant {
        family: "qwen2",
        size: "1.5b",
        n_layers: 28,
        hidden: 1536,
        n_kv_heads: 2,
    },
    SizeVariant {
        family: "qwen2",
        size: "7b",
        n_layers: 28,
        hidden: 3584,
        n_kv_heads: 4,
    },
    SizeVariant {
        family: "qwen2",
        size: "14b",
        n_layers: 48,
        hidden: 5120,
        n_kv_heads: 8,
    },
    SizeVariant {
        family: "llama",
        size: "8b",
        n_layers: 32,
        hidden: 4096,
        n_kv_heads: 8,
    },
    SizeVariant {
        family: "llama",
        size: "70b",
        n_layers: 80,
        hidden: 8192,
        n_kv_heads: 8,
    },
];

#[derive(Debug, PartialEq)]
pub enum LookupVerdict {
    Exact(SizeVariant),
    UnknownSize {
        family: String,
        available: Vec<&'static str>,
    },
    UnknownFamily,
}

pub fn lookup(family: &str, size: Option<&str>) -> LookupVerdict {
    let in_family: Vec<&SizeVariant> = VARIANTS.iter().filter(|v| v.family == family).collect();
    if in_family.is_empty() {
        return LookupVerdict::UnknownFamily;
    }
    let Some(s) = size else {
        // Family-wide lookup with no specific size — return the smallest variant
        // as the canonical "describes the family" representative.
        return LookupVerdict::Exact(*in_family[0]);
    };
    if let Some(v) = in_family.iter().find(|v| v.size == s) {
        return LookupVerdict::Exact(**v);
    }
    LookupVerdict::UnknownSize {
        family: family.into(),
        available: in_family.iter().map(|v| v.size).collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_oracle_size_constraint_filter")?;

    let cases: &[(&str, Option<&str>)] = &[
        ("qwen2", Some("0.5b")),
        ("qwen2", Some("7b")),
        ("qwen2", Some("100b")), // unknown size
        ("llama", None),         // family-wide, no size
        ("exotic", Some("7b")),  // unknown family
    ];

    for (fam, sz) in cases {
        let v = lookup(fam, *sz);
        println!("{fam:>8} {:>6?}  →  {v:?}", sz);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_returns_variant() {
        let v = lookup("qwen2", Some("7b"));
        if let LookupVerdict::Exact(s) = v {
            assert_eq!(s.n_layers, 28);
            assert_eq!(s.hidden, 3584);
            assert_eq!(s.n_kv_heads, 4);
        } else {
            panic!("expected Exact, got {v:?}");
        }
    }

    #[test]
    fn unknown_size_lists_available_options() {
        let v = lookup("qwen2", Some("100b"));
        if let LookupVerdict::UnknownSize { available, .. } = v {
            // Helps the operator pick a valid size on retry.
            assert!(available.contains(&"0.5b"));
            assert!(available.contains(&"7b"));
        } else {
            panic!("expected UnknownSize, got {v:?}");
        }
    }

    #[test]
    fn unknown_family_short_circuits() {
        let v = lookup("exotic", Some("7b"));
        assert_eq!(v, LookupVerdict::UnknownFamily);
    }

    #[test]
    fn family_no_size_returns_smallest_variant() {
        // Family-wide lookup chooses smallest as the canonical representative.
        let v = lookup("qwen2", None);
        if let LookupVerdict::Exact(s) = v {
            assert_eq!(s.size, "0.5b");
        } else {
            panic!("expected Exact, got {v:?}");
        }
    }

    #[test]
    fn n_kv_heads_grows_with_model_size() {
        // Sanity invariant for the variant table — KV heads scale with size.
        // If this regresses someone added a wrong row.
        let small = lookup("qwen2", Some("0.5b"));
        let big = lookup("qwen2", Some("14b"));
        if let (LookupVerdict::Exact(s), LookupVerdict::Exact(b)) = (small, big) {
            assert!(b.n_kv_heads >= s.n_kv_heads);
        } else {
            panic!("variant lookup regressed");
        }
    }
}
