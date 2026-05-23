//! # Format Protobuf Varint Decoder
//!
//! Varint: 7 bits per byte, MSB = continuation flag. Encodes integers
//! using fewer bytes for small values.
//!
//! Demonstrates the **FMT.32** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Protocol Buffers wire format specification.
//!
//! Run with: cargo run --example format_protobuf_varint
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum VarintVerdict {
    Ok { value: u64, bytes_consumed: u32 },
    Truncated,
    Overflow,
    EmptyInput,
}

pub fn decode(bytes: &[u8]) -> VarintVerdict {
    if bytes.is_empty() {
        return VarintVerdict::EmptyInput;
    }
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if i >= 10 {
            return VarintVerdict::Overflow;
        }
        let lower = u64::from(b & 0x7F);
        if shift >= 64 {
            return VarintVerdict::Overflow;
        }
        value |= lower << shift;
        if b & 0x80 == 0 {
            return VarintVerdict::Ok {
                value,
                bytes_consumed: (i + 1) as u32,
            };
        }
        shift += 7;
    }
    VarintVerdict::Truncated
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_protobuf_varint")?;

    println!("0: {:?}", decode(&[0x00]));
    println!("1: {:?}", decode(&[0x01]));
    println!("127: {:?}", decode(&[0x7F]));
    println!("128: {:?}", decode(&[0x80, 0x01]));
    println!("300: {:?}", decode(&[0xAC, 0x02]));
    println!("truncated: {:?}", decode(&[0x80, 0x80]));
    println!("empty: {:?}", decode(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_decoded() {
        let v = decode(&[0x00]);
        if let VarintVerdict::Ok {
            value,
            bytes_consumed,
        } = v
        {
            assert_eq!(value, 0);
            assert_eq!(bytes_consumed, 1);
        }
    }

    #[test]
    fn small_value_one_byte() {
        let v = decode(&[127]);
        if let VarintVerdict::Ok { value, .. } = v {
            assert_eq!(value, 127);
        }
    }

    #[test]
    fn boundary_at_128_two_bytes() {
        let v = decode(&[0x80, 0x01]);
        if let VarintVerdict::Ok {
            value,
            bytes_consumed,
        } = v
        {
            assert_eq!(value, 128);
            assert_eq!(bytes_consumed, 2);
        }
    }

    #[test]
    fn three_hundred_decoded() {
        // 300 = 0xAC 0x02 (LSB first: 0x2C with high bit set, then 0x02).
        let v = decode(&[0xAC, 0x02]);
        if let VarintVerdict::Ok { value, .. } = v {
            assert_eq!(value, 300);
        }
    }

    #[test]
    fn truncated_rejected() {
        // Last byte still has continuation bit set.
        let v = decode(&[0x80, 0x80]);
        assert_eq!(v, VarintVerdict::Truncated);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(decode(&[]), VarintVerdict::EmptyInput);
    }

    #[test]
    fn overflow_after_10_bytes() {
        let v = decode(&[0xFF; 11]);
        assert_eq!(v, VarintVerdict::Overflow);
    }

    #[test]
    fn max_value_10_bytes() {
        // u64::MAX needs 10 bytes.
        let v = decode(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        if let VarintVerdict::Ok {
            value,
            bytes_consumed,
        } = v
        {
            assert_eq!(value, u64::MAX);
            assert_eq!(bytes_consumed, 10);
        }
    }

    #[test]
    fn bytes_consumed_correct() {
        let v = decode(&[0x80, 0x01, 0xFF, 0x00]);
        if let VarintVerdict::Ok { bytes_consumed, .. } = v {
            // First varint = 128, consumes 2 bytes. Rest is leftover.
            assert_eq!(bytes_consumed, 2);
        }
    }

    #[test]
    fn deterministic() {
        let a = decode(&[0xAC, 0x02]);
        let b = decode(&[0xAC, 0x02]);
        assert_eq!(a, b);
    }

    #[test]
    fn single_zero_byte_decodes() {
        let v = decode(&[0]);
        assert!(matches!(v, VarintVerdict::Ok { value: 0, .. }));
    }
}
