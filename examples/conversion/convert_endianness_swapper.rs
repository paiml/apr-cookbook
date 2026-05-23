//! # Conversion Endianness Swapper
//!
//! Tensor data on big-endian hosts (PowerPC, older SPARC) needs byte
//! swap to land in little-endian APR/SafeTensors files. This recipe
//! provides per-dtype swap (no-op for u8, byte-pair swap for u16/i16,
//! 4-byte for f32/u32/i32, 8-byte for f64/u64/i64).
//!
//! Demonstrates the **CONV.13** recipe for PMAT-136 (conversion round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SafeTensors specification (little-endian wire format).
//!
//! Run with: cargo run --example convert_endianness_swapper
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementSize {
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, PartialEq)]
pub enum SwapVerdict {
    Ok { swapped: Vec<u8> },
    LengthNotMultipleOfElement { len: usize, elem: usize },
    EmptyInput,
}

pub fn swap(bytes: &[u8], elem: ElementSize) -> SwapVerdict {
    if bytes.is_empty() {
        return SwapVerdict::EmptyInput;
    }
    let elem_bytes = match elem {
        ElementSize::U8 => 1,
        ElementSize::U16 => 2,
        ElementSize::U32 => 4,
        ElementSize::U64 => 8,
    };
    if bytes.len() % elem_bytes != 0 {
        return SwapVerdict::LengthNotMultipleOfElement {
            len: bytes.len(),
            elem: elem_bytes,
        };
    }
    let mut out = bytes.to_vec();
    for chunk in out.chunks_mut(elem_bytes) {
        chunk.reverse();
    }
    SwapVerdict::Ok { swapped: out }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_endianness_swapper")?;

    let bytes_u32 = [0x12u8, 0x34, 0x56, 0x78];
    println!("u32 swap: {:?}", swap(&bytes_u32, ElementSize::U32));

    let bytes_u16 = [0xAA, 0xBB, 0xCC, 0xDD];
    println!("u16 swap: {:?}", swap(&bytes_u16, ElementSize::U16));

    let bytes_u8 = [0x01, 0x02, 0x03];
    println!("u8 (no-op): {:?}", swap(&bytes_u8, ElementSize::U8));

    println!(
        "misaligned: {:?}",
        swap(&[0x01, 0x02, 0x03], ElementSize::U32)
    );
    println!("empty: {:?}", swap(&[], ElementSize::U32));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn u8_is_noop() {
        let bytes = [0x01, 0x02, 0x03];
        if let SwapVerdict::Ok { swapped } = swap(&bytes, ElementSize::U8) {
            assert_eq!(swapped, bytes.to_vec());
        }
    }

    #[test]
    fn u16_swaps_byte_pairs() {
        let bytes = [0xAA, 0xBB, 0xCC, 0xDD];
        if let SwapVerdict::Ok { swapped } = swap(&bytes, ElementSize::U16) {
            assert_eq!(swapped, vec![0xBB, 0xAA, 0xDD, 0xCC]);
        }
    }

    #[test]
    fn u32_swaps_word_quads() {
        let bytes = [0x12, 0x34, 0x56, 0x78];
        if let SwapVerdict::Ok { swapped } = swap(&bytes, ElementSize::U32) {
            assert_eq!(swapped, vec![0x78, 0x56, 0x34, 0x12]);
        }
    }

    #[test]
    fn u64_swaps_eight_bytes() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        if let SwapVerdict::Ok { swapped } = swap(&bytes, ElementSize::U64) {
            assert_eq!(
                swapped,
                vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
            );
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(swap(&[], ElementSize::U32), SwapVerdict::EmptyInput);
    }

    #[test]
    fn misaligned_input_rejected() {
        let v = swap(&[0x01, 0x02, 0x03], ElementSize::U32);
        assert_eq!(
            v,
            SwapVerdict::LengthNotMultipleOfElement { len: 3, elem: 4 }
        );
    }

    #[test]
    fn round_trip_recovers_original() {
        let original = [0x12u8, 0x34, 0x56, 0x78];
        let SwapVerdict::Ok { swapped: once } = swap(&original, ElementSize::U32) else {
            panic!("first swap failed");
        };
        let SwapVerdict::Ok { swapped: twice } = swap(&once, ElementSize::U32) else {
            panic!("second swap failed");
        };
        assert_eq!(twice, original.to_vec());
    }

    #[test]
    fn multi_element_u32() {
        let bytes = [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD];
        if let SwapVerdict::Ok { swapped } = swap(&bytes, ElementSize::U32) {
            assert_eq!(
                swapped,
                vec![0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA]
            );
        }
    }

    #[test]
    fn misaligned_u16_rejected() {
        let v = swap(&[0x01, 0x02, 0x03], ElementSize::U16);
        assert!(matches!(v, SwapVerdict::LengthNotMultipleOfElement { .. }));
    }

    #[test]
    fn single_byte_for_u8_succeeds() {
        if let SwapVerdict::Ok { swapped } = swap(&[0xFF], ElementSize::U8) {
            assert_eq!(swapped, vec![0xFF]);
        }
    }
}
