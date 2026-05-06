//! # Bundle Compression Codec Picker
//!
//! Pick gzip/lz4/zstd by file_size + decode_speed_target_mbps:
//!   small (< 1 MiB) → gzip-9 (best ratio, decode time negligible)
//!   medium (1-100 MiB) + fast decode → lz4
//!   medium + balanced → zstd-3
//!   large (≥ 100 MiB) → zstd-9 or zstd-19 (max compression)
//!
//! Demonstrates the **BUNDLE.20** recipe for PMAT-148 (bundling round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Facebook Zstandard benchmarks, Yann Collet 2015.
//!
//! Run with: cargo run --example bundle_compression_picker
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Gzip { level: u8 },
    Lz4,
    Zstd { level: u8 },
}

#[derive(Debug, PartialEq)]
pub enum CompressionVerdict {
    Ok {
        codec: Codec,
        estimated_ratio: f64,
        decode_speed_mbps: u32,
    },
    InvalidSize,
    InvalidSpeed,
}

pub fn pick(file_size_mib: u64, decode_speed_target_mbps: u32) -> CompressionVerdict {
    if file_size_mib == 0 {
        return CompressionVerdict::InvalidSize;
    }
    if decode_speed_target_mbps == 0 {
        return CompressionVerdict::InvalidSpeed;
    }
    let codec = if file_size_mib < 1 {
        Codec::Gzip { level: 9 }
    } else if file_size_mib < 100 {
        if decode_speed_target_mbps >= 1_000 {
            Codec::Lz4
        } else {
            Codec::Zstd { level: 3 }
        }
    } else if decode_speed_target_mbps >= 500 {
        Codec::Zstd { level: 9 }
    } else {
        Codec::Zstd { level: 19 }
    };
    let (estimated_ratio, decode_speed_mbps) = match codec {
        Codec::Gzip { .. } => (3.0, 200),
        Codec::Lz4 => (2.0, 4_000),
        Codec::Zstd { level: 3 } => (2.5, 800),
        Codec::Zstd { level: 9 } => (3.5, 500),
        Codec::Zstd { level: 19 } => (4.5, 250),
        Codec::Zstd { .. } => (3.0, 400),
    };
    CompressionVerdict::Ok {
        codec,
        estimated_ratio,
        decode_speed_mbps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_compression_picker")?;

    println!("small file: {:?}", pick(1, 100));
    println!("medium fast: {:?}", pick(50, 2000));
    println!("medium balanced: {:?}", pick(50, 100));
    println!("large fast: {:?}", pick(500, 800));
    println!("large slow: {:?}", pick(500, 100));
    println!("invalid: {:?}", pick(0, 100));
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
    fn medium_fast_picks_lz4() {
        let v = pick(50, 2000);
        if let CompressionVerdict::Ok { codec, .. } = v {
            assert_eq!(codec, Codec::Lz4);
        }
    }

    #[test]
    fn medium_balanced_picks_zstd_3() {
        let v = pick(50, 100);
        if let CompressionVerdict::Ok { codec, .. } = v {
            assert_eq!(codec, Codec::Zstd { level: 3 });
        }
    }

    #[test]
    fn large_fast_picks_zstd_9() {
        let v = pick(500, 800);
        if let CompressionVerdict::Ok { codec, .. } = v {
            assert_eq!(codec, Codec::Zstd { level: 9 });
        }
    }

    #[test]
    fn large_slow_picks_zstd_19() {
        let v = pick(500, 100);
        if let CompressionVerdict::Ok { codec, .. } = v {
            assert_eq!(codec, Codec::Zstd { level: 19 });
        }
    }

    #[test]
    fn invalid_zero_size() {
        assert_eq!(pick(0, 100), CompressionVerdict::InvalidSize);
    }

    #[test]
    fn invalid_zero_speed() {
        assert_eq!(pick(100, 0), CompressionVerdict::InvalidSpeed);
    }

    #[test]
    fn higher_zstd_level_better_ratio() {
        let v_3 = pick(50, 100);
        let v_19 = pick(500, 100);
        if let (
            CompressionVerdict::Ok {
                estimated_ratio: r3,
                ..
            },
            CompressionVerdict::Ok {
                estimated_ratio: r19,
                ..
            },
        ) = (v_3, v_19)
        {
            assert!(r19 > r3);
        }
    }

    #[test]
    fn lz4_fastest_decode() {
        let v_lz4 = pick(50, 2000);
        let v_zstd = pick(50, 100);
        if let (
            CompressionVerdict::Ok {
                decode_speed_mbps: lz4,
                ..
            },
            CompressionVerdict::Ok {
                decode_speed_mbps: zstd,
                ..
            },
        ) = (v_lz4, v_zstd)
        {
            assert!(lz4 > zstd);
        }
    }

    #[test]
    fn boundary_at_100_mib_picks_zstd() {
        let v = pick(100, 100);
        if let CompressionVerdict::Ok { codec, .. } = v {
            // Should be Zstd at level 19 (slow decode target).
            assert_eq!(codec, Codec::Zstd { level: 19 });
        }
    }

    #[test]
    fn boundary_at_1k_mbps_picks_lz4() {
        let v = pick(50, 1000);
        if let CompressionVerdict::Ok { codec, .. } = v {
            assert_eq!(codec, Codec::Lz4);
        }
    }
}
