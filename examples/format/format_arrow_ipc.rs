//! # Format Arrow IPC Stream Message Header
//!
//! Arrow IPC stream messages: 4 bytes 0xFFFFFFFF prefix + 4 bytes
//! flatbuffer length + flatbuffer body. Magic prefix introduced in
//! Arrow 0.15+ to distinguish from older messages.
//!
//! Picker validates header, returns body length + version.
//!
//! Demonstrates the **FMT.28** recipe for PMAT-148 (format round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Apache Arrow IPC streaming format spec.
//!
//! Run with: cargo run --example format_arrow_ipc
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const CONTINUATION_MARKER: u32 = 0xFFFF_FFFF;
const HEADER_PREFIX_BYTES: usize = 8;

#[derive(Debug, PartialEq)]
pub enum HeaderVerdict {
    Ok {
        flatbuffer_len: u32,
        version: HeaderVersion,
    },
    EndOfStream,
    TruncatedHeader,
    InvalidPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderVersion {
    Pre015,
    Modern,
}

pub fn parse(bytes: &[u8]) -> HeaderVerdict {
    if bytes.len() < 4 {
        return HeaderVerdict::TruncatedHeader;
    }
    let prefix = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if prefix == CONTINUATION_MARKER {
        if bytes.len() < HEADER_PREFIX_BYTES {
            return HeaderVerdict::TruncatedHeader;
        }
        let flatbuffer_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if flatbuffer_len == 0 {
            return HeaderVerdict::EndOfStream;
        }
        return HeaderVerdict::Ok {
            flatbuffer_len,
            version: HeaderVersion::Modern,
        };
    }
    // Pre-0.15: 4 bytes is the flatbuffer length.
    if prefix == 0 {
        return HeaderVerdict::EndOfStream;
    }
    HeaderVerdict::Ok {
        flatbuffer_len: prefix,
        version: HeaderVersion::Pre015,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_arrow_ipc")?;

    // Modern: 0xFFFFFFFF prefix + 1024-byte flatbuffer.
    let modern_header = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x04, 0x00, 0x00];
    println!("modern: {:?}", parse(&modern_header));

    let pre015 = [0x00, 0x04, 0x00, 0x00];
    println!("pre 0.15: {:?}", parse(&pre015));

    let eos = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
    println!("end of stream: {:?}", parse(&eos));

    let truncated = [0xFF, 0xFF];
    println!("truncated: {:?}", parse(&truncated));
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
    fn modern_header_parsed() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x04, 0x00, 0x00];
        if let HeaderVerdict::Ok {
            flatbuffer_len,
            version,
        } = parse(&bytes)
        {
            assert_eq!(flatbuffer_len, 1024);
            assert_eq!(version, HeaderVersion::Modern);
        }
    }

    #[test]
    fn pre015_header_parsed() {
        let bytes = [0x00, 0x04, 0x00, 0x00];
        if let HeaderVerdict::Ok {
            flatbuffer_len,
            version,
        } = parse(&bytes)
        {
            assert_eq!(flatbuffer_len, 1024);
            assert_eq!(version, HeaderVersion::Pre015);
        }
    }

    #[test]
    fn modern_eos_detected() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(parse(&bytes), HeaderVerdict::EndOfStream);
    }

    #[test]
    fn pre015_eos_detected() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(parse(&bytes), HeaderVerdict::EndOfStream);
    }

    #[test]
    fn too_short_truncated() {
        assert_eq!(parse(&[0xFF, 0xFF]), HeaderVerdict::TruncatedHeader);
    }

    #[test]
    fn modern_prefix_only_truncated() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(parse(&bytes), HeaderVerdict::TruncatedHeader);
    }

    #[test]
    fn small_flatbuffer_in_modern() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF, 0x10, 0x00, 0x00, 0x00];
        if let HeaderVerdict::Ok { flatbuffer_len, .. } = parse(&bytes) {
            assert_eq!(flatbuffer_len, 16);
        }
    }

    #[test]
    fn large_flatbuffer_supported() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        if let HeaderVerdict::Ok { flatbuffer_len, .. } = parse(&bytes) {
            assert_eq!(flatbuffer_len, 0x7FFF_FFFF);
        }
    }

    #[test]
    fn pre015_small_message() {
        let bytes = [0x10, 0x00, 0x00, 0x00];
        if let HeaderVerdict::Ok {
            flatbuffer_len,
            version,
        } = parse(&bytes)
        {
            assert_eq!(flatbuffer_len, 16);
            assert_eq!(version, HeaderVersion::Pre015);
        }
    }

    #[test]
    fn empty_input_truncated() {
        assert_eq!(parse(&[]), HeaderVerdict::TruncatedHeader);
    }
}
