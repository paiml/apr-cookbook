//! # Data Compression Codec Picker
//!
//! Codecs: None (raw), Lz4 (fast/medium ratio), Zstd1..19 (slow/best
//! ratio). Pick by:
//!
//! - target_throughput_mbps > 1000  → Lz4 or None
//! - target_throughput_mbps 100-1000 → Zstd 3
//! - target_throughput_mbps < 100   → Zstd 9 or 19 (cold archives)
//!
//! Plus: small files (<1 MiB) skip compression to avoid header
//! overhead. This recipe builds the picker.
//!
//! Demonstrates the **DATA.20** recipe for PMAT-135 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Facebook Zstandard benchmark (level vs throughput).
//!
//! Run with: cargo run --example data_compression_codec
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    None,
    Lz4,
    Zstd { level: u8 },
}

#[derive(Debug, PartialEq)]
pub enum CodecVerdict {
    Ok(Codec),
    InvalidThroughput,
    InvalidSize,
}

const SMALL_FILE_THRESHOLD_MIB: u64 = 1;

pub fn pick(file_size_mib: u64, target_throughput_mbps: u32) -> CodecVerdict {
    if target_throughput_mbps == 0 {
        return CodecVerdict::InvalidThroughput;
    }
    if file_size_mib == 0 {
        return CodecVerdict::InvalidSize;
    }
    if file_size_mib < SMALL_FILE_THRESHOLD_MIB {
        return CodecVerdict::Ok(Codec::None);
    }
    let codec = match target_throughput_mbps {
        0..=99 => Codec::Zstd { level: 19 },
        100..=999 => Codec::Zstd { level: 3 },
        1000..=4999 => Codec::Lz4,
        _ => Codec::None,
    };
    CodecVerdict::Ok(codec)
}

pub fn estimated_ratio(codec: Codec) -> f64 {
    match codec {
        Codec::None => 1.0,
        Codec::Lz4 => 2.5,
        Codec::Zstd { level } => match level {
            1..=3 => 3.0,
            4..=9 => 3.5,
            _ => 4.0,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_compression_codec")?;

    for (size, throughput) in [
        (100u64, 50u32),
        (100, 500),
        (100, 2500),
        (100, 10000),
        (0, 500),
        (100, 0),
    ] {
        println!(
            "size={size}MiB tp={throughput}Mbps → {:?}",
            pick(size, throughput)
        );
    }
    println!(
        "ratio Zstd-19: {}",
        estimated_ratio(Codec::Zstd { level: 19 })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn cold_archive_picks_zstd_19() {
        assert_eq!(pick(100, 50), CodecVerdict::Ok(Codec::Zstd { level: 19 }));
    }

    #[test]
    fn medium_throughput_picks_zstd_3() {
        assert_eq!(pick(100, 500), CodecVerdict::Ok(Codec::Zstd { level: 3 }));
    }

    #[test]
    fn high_throughput_picks_lz4() {
        assert_eq!(pick(100, 2500), CodecVerdict::Ok(Codec::Lz4));
    }

    #[test]
    fn extreme_throughput_picks_none() {
        assert_eq!(pick(100, 10_000), CodecVerdict::Ok(Codec::None));
    }

    #[test]
    fn zero_throughput_invalid() {
        assert_eq!(pick(100, 0), CodecVerdict::InvalidThroughput);
    }

    #[test]
    fn zero_size_invalid() {
        assert_eq!(pick(0, 500), CodecVerdict::InvalidSize);
    }

    #[test]
    fn ratio_none_is_one() {
        assert!((estimated_ratio(Codec::None) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ratio_zstd_higher_for_higher_level() {
        let r3 = estimated_ratio(Codec::Zstd { level: 3 });
        let r19 = estimated_ratio(Codec::Zstd { level: 19 });
        assert!(r19 > r3);
    }

    #[test]
    fn ratio_lz4_between_none_and_zstd() {
        let r_lz4 = estimated_ratio(Codec::Lz4);
        let r_none = estimated_ratio(Codec::None);
        let r_zstd = estimated_ratio(Codec::Zstd { level: 3 });
        assert!(r_lz4 > r_none);
        assert!(r_lz4 < r_zstd);
    }

    #[test]
    fn boundary_at_100_picks_zstd_3() {
        // 100 ≤ throughput → Zstd 3 (not Zstd 19).
        assert_eq!(pick(100, 100), CodecVerdict::Ok(Codec::Zstd { level: 3 }));
    }

    #[test]
    fn boundary_at_1000_picks_lz4() {
        assert_eq!(pick(100, 1000), CodecVerdict::Ok(Codec::Lz4));
    }
}
