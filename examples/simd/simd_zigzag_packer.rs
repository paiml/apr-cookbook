//! # SIMD Zigzag Integer Packer
//!
//! Zigzag encoding maps signed → unsigned for varint storage:
//!   encode: (n << 1) ^ (n >> 31)  for i32
//!   decode: (z >> 1) ^ (-(z & 1))
//!
//! Picker validates: input is i32 (zigzag tested), no overflow.
//!
//! Demonstrates the **SIMD.17** recipe for PMAT-147 (simd round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Protocol Buffers wire-format spec § zigzag encoding.
//!
//! Run with: cargo run --example simd_zigzag_packer
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ZigzagVerdict {
    Ok {
        encoded: Vec<u32>,
        decoded: Vec<i32>,
        round_trip_match: bool,
    },
    EmptyInput,
}

pub fn pack_unpack(values: &[i32]) -> ZigzagVerdict {
    if values.is_empty() {
        return ZigzagVerdict::EmptyInput;
    }
    let encoded: Vec<u32> = values.iter().map(|&n| zigzag_encode(n)).collect();
    let decoded: Vec<i32> = encoded.iter().map(|&z| zigzag_decode(z)).collect();
    let round_trip_match = decoded == values;
    ZigzagVerdict::Ok {
        encoded,
        decoded,
        round_trip_match,
    }
}

pub fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

pub fn zigzag_decode(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_zigzag_packer")?;

    println!("typical: {:?}", pack_unpack(&[0, -1, 1, -2, 2, 100, -100]));
    println!("extremes: {:?}", pack_unpack(&[i32::MIN, i32::MAX]));
    println!("empty: {:?}", pack_unpack(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_encodes_to_zero() {
        assert_eq!(zigzag_encode(0), 0);
    }

    #[test]
    fn neg_one_encodes_to_one() {
        assert_eq!(zigzag_encode(-1), 1);
    }

    #[test]
    fn pos_one_encodes_to_two() {
        assert_eq!(zigzag_encode(1), 2);
    }

    #[test]
    fn round_trip_preserves_value() {
        for n in [0, 1, -1, 2, -2, 100, -100, i32::MAX, i32::MIN] {
            assert_eq!(zigzag_decode(zigzag_encode(n)), n);
        }
    }

    #[test]
    fn round_trip_match_flag_true() {
        let v = pack_unpack(&[0, -1, 1, 100]);
        if let ZigzagVerdict::Ok {
            round_trip_match, ..
        } = v
        {
            assert!(round_trip_match);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(pack_unpack(&[]), ZigzagVerdict::EmptyInput);
    }

    #[test]
    fn small_negatives_have_small_encodings() {
        // -1 → 1, -2 → 3, -3 → 5 (small values get small unsigned).
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(-3), 5);
    }

    #[test]
    fn small_positives_have_small_encodings() {
        // 1 → 2, 2 → 4, 3 → 6.
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(2), 4);
        assert_eq!(zigzag_encode(3), 6);
    }

    #[test]
    fn extreme_values_round_trip() {
        let v = pack_unpack(&[i32::MIN, i32::MAX]);
        if let ZigzagVerdict::Ok {
            decoded,
            round_trip_match,
            ..
        } = v
        {
            assert!(round_trip_match);
            assert_eq!(decoded, vec![i32::MIN, i32::MAX]);
        }
    }

    #[test]
    fn encoded_length_matches_input() {
        let v = pack_unpack(&[1, 2, 3, 4, 5]);
        if let ZigzagVerdict::Ok { encoded, .. } = v {
            assert_eq!(encoded.len(), 5);
        }
    }

    #[test]
    fn decode_zero_returns_zero() {
        assert_eq!(zigzag_decode(0), 0);
    }
}
