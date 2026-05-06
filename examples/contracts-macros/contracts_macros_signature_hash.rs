//! # Contracts-Macros Recipe Signature Hash
//!
//! Compute a stable hash of a recipe signature `(name, inputs, output)`
//! for caching keyed lookups. Order-insensitive across input fields,
//! deterministic across runs.
//!
//! Demonstrates the **CMM.61** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo lockfile content-hash convention.
//!
//! Run with: cargo run --example contracts_macros_signature_hash
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SignatureVerdict {
    Ok { hash: String },
    EmptyName,
}

pub fn compute(name: &str, inputs: &[(&str, &str)], output: &str) -> SignatureVerdict {
    if name.trim().is_empty() {
        return SignatureVerdict::EmptyName;
    }
    let mut sorted_inputs: Vec<(&str, &str)> = inputs.to_vec();
    sorted_inputs.sort_by_key(|(k, _)| *k);
    let mut h: u64 = 14_695_981_039_346_656_037;
    h = mix(h, name);
    h = mix(h, "|");
    for (k, v) in &sorted_inputs {
        h = mix(h, k);
        h = mix(h, "=");
        h = mix(h, v);
        h = mix(h, ",");
    }
    h = mix(h, "->");
    h = mix(h, output);
    SignatureVerdict::Ok {
        hash: format!("{h:016x}"),
    }
}

fn mix(mut h: u64, s: &str) -> u64 {
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_signature_hash")?;

    let inputs = [("x", "f64"), ("y", "f64")];
    println!("typical: {:?}", compute("add", &inputs, "f64"));
    println!(
        "reordered same: {:?}",
        compute("add", &[("y", "f64"), ("x", "f64")], "f64")
    );
    println!("empty name: {:?}", compute("", &[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hasher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn order_invariant() {
        let a = compute("f", &[("x", "i32"), ("y", "i32")], "i32");
        let b = compute("f", &[("y", "i32"), ("x", "i32")], "i32");
        assert_eq!(a, b);
    }

    #[test]
    fn name_change_changes_hash() {
        let a = compute("f", &[], "()");
        let b = compute("g", &[], "()");
        assert_ne!(a, b);
    }

    #[test]
    fn input_change_changes_hash() {
        let a = compute("f", &[("x", "i32")], "()");
        let b = compute("f", &[("x", "u32")], "()");
        assert_ne!(a, b);
    }

    #[test]
    fn output_change_changes_hash() {
        let a = compute("f", &[], "i32");
        let b = compute("f", &[], "u32");
        assert_ne!(a, b);
    }

    #[test]
    fn empty_name_rejected() {
        assert_eq!(compute("", &[], ""), SignatureVerdict::EmptyName);
    }

    #[test]
    fn whitespace_name_rejected() {
        assert_eq!(compute("   ", &[], ""), SignatureVerdict::EmptyName);
    }

    #[test]
    fn hash_format_16_hex() {
        let v = compute("f", &[], "()");
        if let SignatureVerdict::Ok { hash } = v {
            assert_eq!(hash.len(), 16);
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn unicode_input_supported() {
        let v = compute("f", &[("café", "i32")], "()");
        assert!(matches!(v, SignatureVerdict::Ok { .. }));
    }

    #[test]
    fn duplicate_input_keys_distinct() {
        // Two entries with same key but different values still hash deterministically.
        let v = compute("f", &[("x", "a"), ("x", "b")], "()");
        assert!(matches!(v, SignatureVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let inputs = [("x", "i32")];
        let a = compute("f", &inputs, "()");
        let b = compute("f", &inputs, "()");
        assert_eq!(a, b);
    }
}
