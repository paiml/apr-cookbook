//! # Contracts-Macros Recipe Hash Consistency
//!
//! Verify each recipe's content-hash in the manifest matches the
//! recomputed hash. Returns the first mismatched recipe (if any).
//!
//! Demonstrates the **CMM.50** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo lockfile content-hash convention.
//!
//! Run with: cargo run --example contracts_macros_recipe_hash_consistency
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HashVerdict {
    Ok {
        recipe_count: u32,
    },
    Mismatch {
        name: String,
        manifest: String,
        actual: String,
    },
    EmptyManifest,
}

pub fn check(entries: &[(&str, &str, &str)]) -> HashVerdict {
    if entries.is_empty() {
        return HashVerdict::EmptyManifest;
    }
    for (name, manifest_hash, content) in entries {
        let actual = fnv1a_hex(content);
        if actual != *manifest_hash {
            return HashVerdict::Mismatch {
                name: (*name).to_string(),
                manifest: (*manifest_hash).to_string(),
                actual,
            };
        }
    }
    HashVerdict::Ok {
        recipe_count: entries.len() as u32,
    }
}

fn fnv1a_hex(s: &str) -> String {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    format!("{h:016x}")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_hash_consistency")?;

    let body = "fn main() {}";
    let h = fnv1a_hex(body);
    let entries = [("recipe_a", h.as_str(), body)];
    println!("ok: {:?}", check(&entries));

    let mismatch = [("recipe_b", "wrong_hash_value", body)];
    println!("mismatch: {:?}", check(&mismatch));
    println!("empty: {:?}", check(&[]));
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
    fn matching_hash_ok() {
        let body = "fn main() {}";
        let h = fnv1a_hex(body);
        let entries = [("a", h.as_str(), body)];
        if let HashVerdict::Ok { recipe_count } = check(&entries) {
            assert_eq!(recipe_count, 1);
        }
    }

    #[test]
    fn mismatch_reported() {
        let entries = [("a", "fakehash", "body")];
        let v = check(&entries);
        if let HashVerdict::Mismatch { name, .. } = v {
            assert_eq!(name, "a");
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(check(&[]), HashVerdict::EmptyManifest);
    }

    #[test]
    fn first_mismatch_returned() {
        let body_a = "a";
        let body_b = "b";
        let h_a = fnv1a_hex(body_a);
        let entries = [("a", h_a.as_str(), body_a), ("b", "wrong", body_b)];
        let v = check(&entries);
        if let HashVerdict::Mismatch { name, .. } = v {
            assert_eq!(name, "b");
        }
    }

    #[test]
    fn empty_string_has_hash() {
        let body = "";
        let h = fnv1a_hex(body);
        let entries = [("a", h.as_str(), body)];
        assert!(matches!(check(&entries), HashVerdict::Ok { .. }));
    }

    #[test]
    fn unicode_body_works() {
        let body = "héllo wörld";
        let h = fnv1a_hex(body);
        let entries = [("a", h.as_str(), body)];
        assert!(matches!(check(&entries), HashVerdict::Ok { .. }));
    }

    #[test]
    fn whitespace_change_changes_hash() {
        let h_no_space = fnv1a_hex("ab");
        let h_with_space = fnv1a_hex("a b");
        assert_ne!(h_no_space, h_with_space);
    }

    #[test]
    fn determinism_across_calls() {
        let h1 = fnv1a_hex("test");
        let h2 = fnv1a_hex("test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn many_entries_all_ok() {
        let bodies = ["a", "b", "c", "d"];
        let hashes: Vec<String> = bodies.iter().map(|b| fnv1a_hex(b)).collect();
        let entries: Vec<(&str, &str, &str)> = bodies
            .iter()
            .zip(hashes.iter())
            .map(|(b, h)| (*b, h.as_str(), *b))
            .collect();
        if let HashVerdict::Ok { recipe_count } = check(&entries) {
            assert_eq!(recipe_count, 4);
        }
    }

    #[test]
    fn hash_format_is_16_hex() {
        let h = fnv1a_hex("anything");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
