//! # Advanced Response Compression Picker
//!
//! Pick compression for inference response body:
//!   <512 B: None (overhead > savings)
//!   512 B - 100 KB: Gzip (good ratio, fast)
//!   ≥ 100 KB: Zstd (better ratio for large bodies)
//! Skip when client doesn't accept compression.
//!
//! Demonstrates the **ADV.36** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTTP Accept-Encoding (RFC 7231) + Cloudflare compression docs.
//!
//! Run with: cargo run --example adv_response_compression_picker
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Gzip,
    Zstd,
}

#[derive(Debug, PartialEq)]
pub enum CompressionVerdict {
    Pick { algo: Compression },
    InvalidSize,
}

pub fn pick(body_size_bytes: i64, client_accepts: &[&str]) -> CompressionVerdict {
    if body_size_bytes < 0 {
        return CompressionVerdict::InvalidSize;
    }
    let size = body_size_bytes as u64;
    let accepts_gzip = client_accepts
        .iter()
        .any(|s| s.eq_ignore_ascii_case("gzip"));
    let accepts_zstd = client_accepts
        .iter()
        .any(|s| s.eq_ignore_ascii_case("zstd"));
    if size < 512 {
        return CompressionVerdict::Pick {
            algo: Compression::None,
        };
    }
    let preferred = if size >= 100_000 {
        Compression::Zstd
    } else {
        Compression::Gzip
    };
    let algo = match preferred {
        Compression::Zstd if accepts_zstd => Compression::Zstd,
        Compression::Zstd if accepts_gzip => Compression::Gzip,
        Compression::Gzip if accepts_gzip => Compression::Gzip,
        Compression::Zstd | Compression::Gzip | Compression::None => Compression::None,
    };
    CompressionVerdict::Pick { algo }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_response_compression_picker")?;

    println!("tiny: {:?}", pick(100, &["gzip"]));
    println!("medium gzip: {:?}", pick(10_000, &["gzip"]));
    println!("large zstd: {:?}", pick(500_000, &["zstd"]));
    println!("large gzip fallback: {:?}", pick(500_000, &["gzip"]));
    println!("nothing accepted: {:?}", pick(500_000, &[]));
    println!("invalid: {:?}", pick(-1, &["gzip"]));
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
    fn tiny_skips_compression() {
        let v = pick(100, &["gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::None);
        }
    }

    #[test]
    fn medium_picks_gzip() {
        let v = pick(10_000, &["gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Gzip);
        }
    }

    #[test]
    fn large_picks_zstd_when_accepted() {
        let v = pick(500_000, &["zstd"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Zstd);
        }
    }

    #[test]
    fn large_falls_back_to_gzip() {
        let v = pick(500_000, &["gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Gzip);
        }
    }

    #[test]
    fn no_accept_no_compression() {
        let v = pick(500_000, &[]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::None);
        }
    }

    #[test]
    fn negative_size_rejected() {
        assert_eq!(pick(-1, &["gzip"]), CompressionVerdict::InvalidSize);
    }

    #[test]
    fn case_insensitive_accept() {
        let v = pick(10_000, &["GZIP"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Gzip);
        }
    }

    #[test]
    fn boundary_at_512_uses_compression() {
        let v = pick(512, &["gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Gzip);
        }
    }

    #[test]
    fn boundary_at_100k_uses_zstd() {
        let v = pick(100_000, &["zstd", "gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::Zstd);
        }
    }

    #[test]
    fn zero_size_no_compression() {
        let v = pick(0, &["gzip"]);
        if let CompressionVerdict::Pick { algo } = v {
            assert_eq!(algo, Compression::None);
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(10_000, &["gzip"]);
        let b = pick(10_000, &["gzip"]);
        assert_eq!(a, b);
    }
}
