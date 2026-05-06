//! # Format Endianness Detector
//!
//! Tensor formats may be little-endian (x86, ARM, GGUF) or big-endian
//! (PowerPC, network byte order, ONNX). Cross-platform read needs
//! per-tensor byte swapping. This recipe builds the detector + swap
//! decision (when reading a foreign-endian payload).
//!
//! Demonstrates the **FMT.20** recipe for PMAT-130 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: IEEE 754-2008 + RFC 1700 (network byte order).
//!
//! Run with: cargo run --example format_endianness_detector
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Debug, PartialEq)]
pub enum DetectVerdict {
    Ok(Endianness),
    InsufficientBytes { need: usize, got: usize },
    Ambiguous,
}

const MAGIC_LE: &[u8] = &[0x78, 0x56, 0x34, 0x12]; // 0x12345678 LE
const MAGIC_BE: &[u8] = &[0x12, 0x34, 0x56, 0x78];

pub fn detect_from_magic(bytes: &[u8]) -> DetectVerdict {
    if bytes.len() < 4 {
        return DetectVerdict::InsufficientBytes {
            need: 4,
            got: bytes.len(),
        };
    }
    let prefix = &bytes[..4];
    if prefix == MAGIC_LE {
        DetectVerdict::Ok(Endianness::Little)
    } else if prefix == MAGIC_BE {
        DetectVerdict::Ok(Endianness::Big)
    } else {
        DetectVerdict::Ambiguous
    }
}

#[derive(Debug, PartialEq)]
pub enum SwapVerdict {
    NoSwapNeeded,
    SwapBytes { width: u8 },
    InvalidWidth,
}

const HOST_ENDIANNESS: Endianness = if cfg!(target_endian = "big") {
    Endianness::Big
} else {
    Endianness::Little
};

pub fn swap_decision(payload_endian: Endianness, type_width_bytes: u8) -> SwapVerdict {
    if !matches!(type_width_bytes, 1 | 2 | 4 | 8) {
        return SwapVerdict::InvalidWidth;
    }
    if type_width_bytes == 1 {
        // Single-byte types are endian-invariant.
        return SwapVerdict::NoSwapNeeded;
    }
    if payload_endian == HOST_ENDIANNESS {
        SwapVerdict::NoSwapNeeded
    } else {
        SwapVerdict::SwapBytes {
            width: type_width_bytes,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_endianness_detector")?;

    println!("LE magic: {:?}", detect_from_magic(MAGIC_LE));
    println!("BE magic: {:?}", detect_from_magic(MAGIC_BE));
    println!(
        "ambiguous: {:?}",
        detect_from_magic(&[0xff, 0xee, 0xdd, 0xcc])
    );
    println!("short: {:?}", detect_from_magic(&[0x78]));

    for w in [1u8, 2, 4, 8, 3] {
        println!("swap LE w={w}: {:?}", swap_decision(Endianness::Little, w));
        println!("swap BE w={w}: {:?}", swap_decision(Endianness::Big, w));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn le_magic_detected() {
        assert_eq!(
            detect_from_magic(&[0x78, 0x56, 0x34, 0x12]),
            DetectVerdict::Ok(Endianness::Little)
        );
    }

    #[test]
    fn be_magic_detected() {
        assert_eq!(
            detect_from_magic(&[0x12, 0x34, 0x56, 0x78]),
            DetectVerdict::Ok(Endianness::Big)
        );
    }

    #[test]
    fn unknown_magic_ambiguous() {
        assert_eq!(
            detect_from_magic(&[0xff, 0xee, 0xdd, 0xcc]),
            DetectVerdict::Ambiguous
        );
    }

    #[test]
    fn short_input_rejected() {
        let v = detect_from_magic(&[0x12]);
        assert!(matches!(
            v,
            DetectVerdict::InsufficientBytes { need: 4, got: 1 }
        ));
    }

    #[test]
    fn swap_byte_invariant_for_width_1() {
        assert_eq!(
            swap_decision(Endianness::Little, 1),
            SwapVerdict::NoSwapNeeded
        );
        assert_eq!(swap_decision(Endianness::Big, 1), SwapVerdict::NoSwapNeeded);
    }

    #[test]
    fn swap_invalid_width_rejected() {
        // Only 1, 2, 4, 8 are valid widths.
        assert_eq!(
            swap_decision(Endianness::Little, 3),
            SwapVerdict::InvalidWidth
        );
        assert_eq!(
            swap_decision(Endianness::Little, 0),
            SwapVerdict::InvalidWidth
        );
    }

    #[test]
    fn matching_endianness_no_swap() {
        let v = swap_decision(HOST_ENDIANNESS, 4);
        assert_eq!(v, SwapVerdict::NoSwapNeeded);
    }

    #[test]
    fn opposite_endianness_swaps() {
        let opposite = if HOST_ENDIANNESS == Endianness::Little {
            Endianness::Big
        } else {
            Endianness::Little
        };
        let v = swap_decision(opposite, 4);
        assert!(matches!(v, SwapVerdict::SwapBytes { width: 4 }));
    }

    #[test]
    fn empty_bytes_rejected() {
        assert!(matches!(
            detect_from_magic(&[]),
            DetectVerdict::InsufficientBytes { got: 0, .. }
        ));
    }

    #[test]
    fn extra_bytes_after_magic_ignored() {
        // Only first 4 bytes matter for detection.
        let v = detect_from_magic(&[0x78, 0x56, 0x34, 0x12, 0xff, 0xff]);
        assert_eq!(v, DetectVerdict::Ok(Endianness::Little));
    }
}
