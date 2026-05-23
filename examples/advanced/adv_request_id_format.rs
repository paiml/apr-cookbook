//! # Advanced Request-ID Format
//!
//! Generate canonical request IDs: `{prefix}-{epoch_ms}-{seq}-{rand}`.
//! This recipe is the format spec/validator: parse a generated ID back
//! and verify each part is well-formed.
//!
//! Demonstrates the **ADV.35** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS request-ID convention (X-Amzn-RequestId).
//!
//! Run with: cargo run --example adv_request_id_format
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IdVerdict {
    Ok {
        prefix: String,
        epoch_ms: u64,
        seq: u32,
        rand_hex_len: u32,
    },
    BadPartCount,
    InvalidPrefix,
    InvalidEpoch,
    InvalidSeq,
    InvalidRand,
}

pub fn parse(id: &str) -> IdVerdict {
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() != 4 {
        return IdVerdict::BadPartCount;
    }
    let prefix = parts[0];
    if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_alphanumeric()) {
        return IdVerdict::InvalidPrefix;
    }
    let Ok(epoch_ms) = parts[1].parse::<u64>() else {
        return IdVerdict::InvalidEpoch;
    };
    let Ok(seq) = parts[2].parse::<u32>() else {
        return IdVerdict::InvalidSeq;
    };
    let rand = parts[3];
    if rand.is_empty() || !rand.chars().all(|c| c.is_ascii_hexdigit()) {
        return IdVerdict::InvalidRand;
    }
    IdVerdict::Ok {
        prefix: prefix.to_string(),
        epoch_ms,
        seq,
        rand_hex_len: rand.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_request_id_format")?;

    println!("ok: {:?}", parse("req-1700000000000-1-deadbeef"));
    println!("bad count: {:?}", parse("req-only"));
    println!("bad prefix: {:?}", parse("re q-1-1-aa"));
    println!("bad epoch: {:?}", parse("req-NOTNUM-1-aa"));
    println!("bad rand: {:?}", parse("req-1-1-NOT_HEX"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_id_parses() {
        let v = parse("req-1700000000000-42-deadbeef");
        if let IdVerdict::Ok {
            prefix,
            epoch_ms,
            seq,
            rand_hex_len,
        } = v
        {
            assert_eq!(prefix, "req");
            assert_eq!(epoch_ms, 1_700_000_000_000);
            assert_eq!(seq, 42);
            assert_eq!(rand_hex_len, 8);
        }
    }

    #[test]
    fn missing_parts_rejected() {
        assert_eq!(parse("req-only"), IdVerdict::BadPartCount);
    }

    #[test]
    fn five_parts_rejected() {
        assert_eq!(parse("a-b-c-d-e"), IdVerdict::BadPartCount);
    }

    #[test]
    fn empty_prefix_rejected() {
        assert_eq!(parse("-1-1-aa"), IdVerdict::InvalidPrefix);
    }

    #[test]
    fn special_chars_in_prefix_rejected() {
        assert_eq!(parse("re@q-1-1-aa"), IdVerdict::InvalidPrefix);
    }

    #[test]
    fn negative_epoch_rejected() {
        assert_eq!(parse("req--1-1-aa"), IdVerdict::BadPartCount);
    }

    #[test]
    fn nan_epoch_rejected() {
        assert_eq!(parse("req-NOTNUM-1-aa"), IdVerdict::InvalidEpoch);
    }

    #[test]
    fn nan_seq_rejected() {
        assert_eq!(parse("req-1-NOTNUM-aa"), IdVerdict::InvalidSeq);
    }

    #[test]
    fn non_hex_rand_rejected() {
        assert_eq!(parse("req-1-1-NOTHEX"), IdVerdict::InvalidRand);
    }

    #[test]
    fn empty_rand_rejected() {
        assert_eq!(parse("req-1-1-"), IdVerdict::InvalidRand);
    }

    #[test]
    fn upper_hex_accepted() {
        let v = parse("req-1-1-DEADBEEF");
        assert!(matches!(v, IdVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let a = parse("req-1-1-aa");
        let b = parse("req-1-1-aa");
        assert_eq!(a, b);
    }
}
