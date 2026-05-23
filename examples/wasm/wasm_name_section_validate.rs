//! # WASM Name Section Validate
//!
//! Validate the `name` custom section: module name + function names
//! must each be ≤ 256 chars, valid UTF-8, no embedded NULs. Returns
//! sorted offending names.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly name-section spec; LLD wasm-ld name-mangling
//!  conventions.
//!
//! Run with: cargo run --example wasm_name_section_validate
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NameVerdict {
    Ok {
        offending_names: Vec<String>,
        valid_count: u32,
    },
    InvalidConfig,
}

pub fn validate(names: &[&str]) -> NameVerdict {
    if names.is_empty() {
        return NameVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = names
        .iter()
        .filter(|n| !is_valid(n))
        .map(|n| (*n).to_string())
        .collect();
    offenders.sort();
    let valid = names.iter().filter(|n| is_valid(n)).count() as u32;
    NameVerdict::Ok {
        offending_names: offenders,
        valid_count: valid,
    }
}

fn is_valid(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if name.chars().count() > 256 {
        return false;
    }
    if name.contains('\0') {
        return false;
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_name_section_validate")?;

    println!("ok: {:?}", validate(&["main", "init"]));
    println!("with-nul: {:?}", validate(&["bad\0name"]));
    println!("empty: {:?}", validate(&[""]));
    println!("invalid: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_name_no_offender() {
        let v = validate(&["main"]);
        if let NameVerdict::Ok {
            offending_names, ..
        } = v
        {
            assert!(offending_names.is_empty());
        }
    }

    #[test]
    fn empty_name_offender() {
        let v = validate(&[""]);
        if let NameVerdict::Ok {
            offending_names, ..
        } = v
        {
            assert_eq!(offending_names, vec!["".to_string()]);
        }
    }

    #[test]
    fn nul_byte_offender() {
        let v = validate(&["bad\0name"]);
        if let NameVerdict::Ok {
            offending_names, ..
        } = v
        {
            assert_eq!(offending_names, vec!["bad\0name".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(validate(&[]), NameVerdict::InvalidConfig);
    }

    #[test]
    fn very_long_name_offender() {
        let long: String = "x".repeat(257);
        let v = validate(&[long.as_str()]);
        if let NameVerdict::Ok {
            offending_names, ..
        } = v
        {
            assert_eq!(offending_names.len(), 1);
        }
    }

    #[test]
    fn boundary_256_chars_valid() {
        let s: String = "x".repeat(256);
        let v = validate(&[s.as_str()]);
        if let NameVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&["main"]);
        let r2 = validate(&["main"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = validate(&["zeta\0", "alpha\0"]);
        if let NameVerdict::Ok {
            offending_names, ..
        } = v
        {
            for w in offending_names.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn valid_count_correct() {
        let v = validate(&["main", "", "init", "x\0"]);
        if let NameVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 2);
        }
    }

    #[test]
    fn unicode_name_valid() {
        let v = validate(&["café_func"]);
        if let NameVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 1);
        }
    }

    #[test]
    fn many_names_handled() {
        let names: Vec<&str> = (0..30).map(|_| "func").collect();
        let v = validate(&names);
        if let NameVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 30);
        }
    }

    #[test]
    fn mixed_valid_invalid() {
        let v = validate(&["good", "", "good2"]);
        if let NameVerdict::Ok {
            offending_names,
            valid_count,
        } = v
        {
            assert_eq!(offending_names.len(), 1);
            assert_eq!(valid_count, 2);
        }
    }
}
