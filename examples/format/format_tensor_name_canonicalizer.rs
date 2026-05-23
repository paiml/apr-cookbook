//! # Format Tensor Name Canonicalizer
//!
//! Cross-framework tensor names diverge: PyTorch `model.layers.0.attn.q_proj.weight`,
//! TensorFlow `model/layers/0/attn/q_proj/kernel:0`, GGUF `blk.0.attn_q.weight`.
//! Canonical form (this recipe): all-lowercase, dot-separated, no
//! ":suffix", no `/` separators. Returns the canonical key + lossy
//! flag if information was dropped.
//!
//! Demonstrates the **FMT.26** recipe for PMAT-136 (format round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace Transformers naming convention.
//!
//! Run with: cargo run --example format_tensor_name_canonicalizer
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CanonVerdict {
    Ok {
        canonical: String,
        dropped_suffix: bool,
    },
    EmptyName,
    LeadingDotOrSlash,
    TrailingDotOrSlash,
}

pub fn canonicalize(name: &str) -> CanonVerdict {
    if name.is_empty() {
        return CanonVerdict::EmptyName;
    }
    if name.starts_with('.') || name.starts_with('/') {
        return CanonVerdict::LeadingDotOrSlash;
    }
    if name.ends_with('.') || name.ends_with('/') {
        return CanonVerdict::TrailingDotOrSlash;
    }
    let mut dropped = false;
    let stripped = if let Some(idx) = name.rfind(':') {
        dropped = true;
        &name[..idx]
    } else {
        name
    };
    let lowered = stripped.to_ascii_lowercase();
    let canonical = lowered.replace('/', ".");
    CanonVerdict::Ok {
        canonical,
        dropped_suffix: dropped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_tensor_name_canonicalizer")?;

    let names = [
        "model.layers.0.attn.q_proj.weight",
        "model/layers/0/attn/q_proj/kernel:0",
        "Model.Embeddings.Token",
        "BLK.0.ATTN_Q.WEIGHT",
        "",
        ".leading-dot",
        "trailing-slash/",
    ];
    for n in names {
        println!("{n} → {:?}", canonicalize(n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn already_canonical_passes_through() {
        let v = canonicalize("model.layers.0.attn.q_proj.weight");
        if let CanonVerdict::Ok {
            canonical,
            dropped_suffix,
        } = v
        {
            assert_eq!(canonical, "model.layers.0.attn.q_proj.weight");
            assert!(!dropped_suffix);
        }
    }

    #[test]
    fn slash_replaced_with_dot() {
        let v = canonicalize("model/layers/0/attn");
        if let CanonVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical, "model.layers.0.attn");
        }
    }

    #[test]
    fn colon_suffix_stripped() {
        let v = canonicalize("model.weight:0");
        if let CanonVerdict::Ok {
            canonical,
            dropped_suffix,
        } = v
        {
            assert_eq!(canonical, "model.weight");
            assert!(dropped_suffix);
        }
    }

    #[test]
    fn uppercase_lowered() {
        let v = canonicalize("Model.Embeddings.TOKEN");
        if let CanonVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical, "model.embeddings.token");
        }
    }

    #[test]
    fn full_tf_name_canonicalized() {
        let v = canonicalize("model/layers/0/attn/q_proj/kernel:0");
        if let CanonVerdict::Ok {
            canonical,
            dropped_suffix,
        } = v
        {
            assert_eq!(canonical, "model.layers.0.attn.q_proj.kernel");
            assert!(dropped_suffix);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(canonicalize(""), CanonVerdict::EmptyName);
    }

    #[test]
    fn leading_dot_rejected() {
        assert_eq!(canonicalize(".bad"), CanonVerdict::LeadingDotOrSlash);
    }

    #[test]
    fn leading_slash_rejected() {
        assert_eq!(canonicalize("/bad"), CanonVerdict::LeadingDotOrSlash);
    }

    #[test]
    fn trailing_dot_rejected() {
        assert_eq!(canonicalize("bad."), CanonVerdict::TrailingDotOrSlash);
    }

    #[test]
    fn trailing_slash_rejected() {
        assert_eq!(canonicalize("bad/"), CanonVerdict::TrailingDotOrSlash);
    }

    #[test]
    fn underscore_preserved() {
        let v = canonicalize("q_proj.weight");
        if let CanonVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical, "q_proj.weight");
        }
    }

    #[test]
    fn multiple_colons_only_strip_after_last() {
        // Only the suffix ":N" is stripped (after last colon).
        let v = canonicalize("a:b:c");
        if let CanonVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical, "a:b");
        }
    }
}
