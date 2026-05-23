//! # Format ZIP End-of-Central-Directory (EOCD) Locator
//!
//! ZIP file's EOCD record is at the end (variable position due to
//! comment). Locator scans backward from end-of-file looking for
//! signature `PK\x05\x06`. Recipe: validate offset + parse central
//! directory size + entry count.
//!
//! Demonstrates the **FMT.27** recipe for PMAT-148 (format round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: APPNOTE.TXT (PKWARE ZIP File Format Specification).
//!
//! Run with: cargo run --example format_zip_eocd
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const EOCD_SIGNATURE: [u8; 4] = [0x50, 0x4B, 0x05, 0x06];
const EOCD_MIN_SIZE: usize = 22;
const MAX_COMMENT_SIZE: usize = 65_535;

#[derive(Debug, PartialEq)]
pub enum EocdVerdict {
    Ok {
        eocd_offset: usize,
        cd_offset: u32,
        cd_size: u32,
        entry_count: u16,
    },
    NotFound,
    FileTooSmall,
    InvalidSignature,
}

pub fn locate(file_bytes: &[u8]) -> EocdVerdict {
    if file_bytes.len() < EOCD_MIN_SIZE {
        return EocdVerdict::FileTooSmall;
    }
    let n = file_bytes.len();
    let scan_start = n.saturating_sub(EOCD_MIN_SIZE + MAX_COMMENT_SIZE);
    let max_eocd = n.saturating_sub(EOCD_MIN_SIZE);
    let mut found_offset: Option<usize> = None;
    for off in (scan_start..=max_eocd).rev() {
        if file_bytes[off..off + 4] == EOCD_SIGNATURE {
            found_offset = Some(off);
            break;
        }
    }
    let Some(off) = found_offset else {
        return EocdVerdict::NotFound;
    };
    if off + EOCD_MIN_SIZE > n {
        return EocdVerdict::InvalidSignature;
    }
    let entry_count = u16::from_le_bytes([file_bytes[off + 10], file_bytes[off + 11]]);
    let cd_size = u32::from_le_bytes([
        file_bytes[off + 12],
        file_bytes[off + 13],
        file_bytes[off + 14],
        file_bytes[off + 15],
    ]);
    let cd_offset = u32::from_le_bytes([
        file_bytes[off + 16],
        file_bytes[off + 17],
        file_bytes[off + 18],
        file_bytes[off + 19],
    ]);
    EocdVerdict::Ok {
        eocd_offset: off,
        cd_offset,
        cd_size,
        entry_count,
    }
}

fn build_minimal_eocd_zip() -> Vec<u8> {
    // Minimal valid EOCD: 22 bytes total. EOCD_signature + zero-CD record.
    let mut bytes = vec![0u8; EOCD_MIN_SIZE];
    bytes[0..4].copy_from_slice(&EOCD_SIGNATURE);
    // disk = 0, disk_with_cd = 0, entries_on_disk = 0, total_entries = 0,
    // cd_size = 0, cd_offset = 0, comment_len = 0.
    bytes
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_zip_eocd")?;

    let zip = build_minimal_eocd_zip();
    println!("minimal: {:?}", locate(&zip));
    println!("too small: {:?}", locate(&[0u8; 5]));
    let no_sig = vec![0u8; 100];
    println!("no signature: {:?}", locate(&no_sig));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn minimal_eocd_located() {
        let zip = build_minimal_eocd_zip();
        let v = locate(&zip);
        assert!(matches!(v, EocdVerdict::Ok { .. }));
    }

    #[test]
    fn too_small_rejected() {
        assert_eq!(locate(&[0u8; 5]), EocdVerdict::FileTooSmall);
    }

    #[test]
    fn no_signature_returns_not_found() {
        let v = locate(&vec![0u8; 100]);
        assert_eq!(v, EocdVerdict::NotFound);
    }

    #[test]
    fn eocd_offset_correct_for_minimal() {
        let zip = build_minimal_eocd_zip();
        if let EocdVerdict::Ok { eocd_offset, .. } = locate(&zip) {
            assert_eq!(eocd_offset, 0);
        }
    }

    #[test]
    fn entry_count_zero_in_minimal() {
        let zip = build_minimal_eocd_zip();
        if let EocdVerdict::Ok { entry_count, .. } = locate(&zip) {
            assert_eq!(entry_count, 0);
        }
    }

    #[test]
    fn eocd_at_end_of_file() {
        let mut zip = vec![0u8; 500];
        let off = zip.len() - EOCD_MIN_SIZE;
        zip[off..off + 4].copy_from_slice(&EOCD_SIGNATURE);
        let v = locate(&zip);
        if let EocdVerdict::Ok { eocd_offset, .. } = v {
            assert_eq!(eocd_offset, off);
        }
    }

    #[test]
    fn eocd_with_comment() {
        // EOCD followed by 50-byte comment.
        let mut zip = vec![0u8; 500];
        let eocd_off = zip.len() - EOCD_MIN_SIZE - 50;
        zip[eocd_off..eocd_off + 4].copy_from_slice(&EOCD_SIGNATURE);
        let v = locate(&zip);
        if let EocdVerdict::Ok { eocd_offset, .. } = v {
            assert_eq!(eocd_offset, eocd_off);
        }
    }

    #[test]
    fn cd_size_parsed_correctly() {
        let mut zip = build_minimal_eocd_zip();
        // Set cd_size to 0xDEADBEEF.
        zip[12..16].copy_from_slice(&[0xEF, 0xBE, 0xAD, 0xDE]);
        if let EocdVerdict::Ok { cd_size, .. } = locate(&zip) {
            assert_eq!(cd_size, 0xDEADBEEF);
        }
    }

    #[test]
    fn finds_last_eocd_when_multiple() {
        // Backwards scan should find the LAST EOCD.
        let mut zip = vec![0u8; 1000];
        zip[100..104].copy_from_slice(&EOCD_SIGNATURE);
        let last = zip.len() - EOCD_MIN_SIZE;
        zip[last..last + 4].copy_from_slice(&EOCD_SIGNATURE);
        if let EocdVerdict::Ok { eocd_offset, .. } = locate(&zip) {
            assert_eq!(eocd_offset, last);
        }
    }

    #[test]
    fn at_minimum_size_works() {
        let zip = build_minimal_eocd_zip();
        assert_eq!(zip.len(), EOCD_MIN_SIZE);
        assert!(matches!(locate(&zip), EocdVerdict::Ok { .. }));
    }
}
