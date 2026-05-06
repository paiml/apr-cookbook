//! # Registry Content-Hash Pin Enforcer
//!
//! Registry pulls SHOULD pin content by sha256 hash for
//! reproducibility. Allowed hash algorithms: sha256 (64 hex chars),
//! sha512 (128 hex chars), blake3 (64 hex chars). Length + alphabet
//! validation. This recipe builds the enforcer + pin-style classifier.
//!
//! Demonstrates the **REG.9** recipe for PMAT-129 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST FIPS 180-4 (SHA), BLAKE3 spec.
//!
//! Run with: cargo run --example registry_hash_pin_enforcer
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgo {
    Sha256,
    Sha512,
    Blake3,
}

#[derive(Debug, PartialEq)]
pub enum PinVerdict {
    Ok {
        algo: HashAlgo,
        hex: String,
    },
    MissingAlgoPrefix,
    UnknownAlgo {
        algo: String,
    },
    BadHexLength {
        algo: HashAlgo,
        got: usize,
        expected: usize,
    },
    NonHexCharacter,
}

pub fn parse_pin(pin: &str) -> PinVerdict {
    let Some((algo_str, hex)) = pin.split_once(':') else {
        return PinVerdict::MissingAlgoPrefix;
    };
    let algo = match algo_str {
        "sha256" => HashAlgo::Sha256,
        "sha512" => HashAlgo::Sha512,
        "blake3" => HashAlgo::Blake3,
        _ => {
            return PinVerdict::UnknownAlgo {
                algo: algo_str.into(),
            };
        }
    };
    let expected_len = match algo {
        HashAlgo::Sha256 | HashAlgo::Blake3 => 64,
        HashAlgo::Sha512 => 128,
    };
    if hex.len() != expected_len {
        return PinVerdict::BadHexLength {
            algo,
            got: hex.len(),
            expected: expected_len,
        };
    }
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return PinVerdict::NonHexCharacter;
    }
    PinVerdict::Ok {
        algo,
        hex: hex.into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinStyle {
    Strict,    // exact hash pin
    SemverTag, // version pin only
    Float,     // "latest" or alias
}

pub fn classify_pin_style(reference: &str) -> PinStyle {
    if reference.contains(':') && reference.starts_with("sha256:")
        || reference.starts_with("sha512:")
        || reference.starts_with("blake3:")
    {
        return PinStyle::Strict;
    }
    let core = reference.split_once('@').map_or(reference, |(_, t)| t);
    if core.split('.').count() == 3 && core.split('.').all(|p| p.parse::<u32>().is_ok()) {
        return PinStyle::SemverTag;
    }
    PinStyle::Float
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_hash_pin_enforcer")?;

    for pin in [
        "sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "sha512:abc",
        "md5:0123456789abcdef0123456789abcdef",
        "no-prefix",
        "sha256:zzzz",
    ] {
        println!("{pin:<70}  →  {:?}", parse_pin(pin));
    }

    for r in ["model@1.2.3", "model@latest", "plain"] {
        println!("{r}  →  {:?}", classify_pin_style(r));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforcer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_sha256_pin_parses() {
        let pin = format!("sha256:{}", "a".repeat(64));
        let v = parse_pin(&pin);
        assert!(matches!(
            v,
            PinVerdict::Ok {
                algo: HashAlgo::Sha256,
                ..
            }
        ));
    }

    #[test]
    fn sha512_pin_parses() {
        let pin = format!("sha512:{}", "b".repeat(128));
        let v = parse_pin(&pin);
        assert!(matches!(
            v,
            PinVerdict::Ok {
                algo: HashAlgo::Sha512,
                ..
            }
        ));
    }

    #[test]
    fn blake3_pin_parses() {
        let pin = format!("blake3:{}", "c".repeat(64));
        let v = parse_pin(&pin);
        assert!(matches!(
            v,
            PinVerdict::Ok {
                algo: HashAlgo::Blake3,
                ..
            }
        ));
    }

    #[test]
    fn missing_prefix_rejected() {
        assert_eq!(parse_pin("no-prefix"), PinVerdict::MissingAlgoPrefix);
    }

    #[test]
    fn unknown_algo_rejected() {
        let pin = format!("md5:{}", "0".repeat(32));
        let v = parse_pin(&pin);
        assert!(matches!(v, PinVerdict::UnknownAlgo { .. }));
    }

    #[test]
    fn wrong_length_rejected() {
        // sha256 expects 64; provide 32.
        let pin = format!("sha256:{}", "a".repeat(32));
        let v = parse_pin(&pin);
        assert!(matches!(
            v,
            PinVerdict::BadHexLength {
                expected: 64,
                got: 32,
                ..
            }
        ));
    }

    #[test]
    fn non_hex_rejected() {
        let pin = format!("sha256:{}", "z".repeat(64));
        assert_eq!(parse_pin(&pin), PinVerdict::NonHexCharacter);
    }

    #[test]
    fn classify_strict_for_hash_pin() {
        let pin = format!("sha256:{}", "a".repeat(64));
        assert_eq!(classify_pin_style(&pin), PinStyle::Strict);
    }

    #[test]
    fn classify_semver_for_dotted_version() {
        assert_eq!(classify_pin_style("model@1.2.3"), PinStyle::SemverTag);
    }

    #[test]
    fn classify_float_for_latest() {
        assert_eq!(classify_pin_style("model@latest"), PinStyle::Float);
        assert_eq!(classify_pin_style("plain-name"), PinStyle::Float);
    }
}
