//! # apr debug — `--strings` ASCII Extractor
//!
//! `apr debug --strings <FILE>` extracts ASCII strings of length ≥ 4
//! from the file (mimics the GNU `strings` utility for forensics on
//! tokenizer vocabs and metadata). This recipe builds the extractor
//! and asserts the contract: minimum length is configurable, embedded
//! NUL bytes terminate strings, runs longer than printable-character
//! threshold get included, runs shorter are filtered.
//!
//! Demonstrates the **DEBUG.4** recipe for PMAT-101 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEBUG-001 + GNU strings(1)
//!
//! Run with: cargo run --example cli_debug_string_extractor
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_MIN_LEN: usize = 4;

pub fn extract_strings(bytes: &[u8], min_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for &b in bytes {
        if (0x20..0x7f).contains(&b) {
            current.push(b as char);
        } else {
            if current.len() >= min_len {
                out.push(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= min_len {
        out.push(current);
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_string_extractor")?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"APR2");
    bytes.push(0); // NUL
    bytes.extend_from_slice(b"<pad>");
    bytes.push(0);
    bytes.extend_from_slice(b"hello");
    bytes.push(0xff); // non-ASCII
    bytes.extend_from_slice(b"x"); // too short, filtered
    bytes.push(0);
    bytes.extend_from_slice(b"world");

    let strings = extract_strings(&bytes, DEFAULT_MIN_LEN);
    for s in &strings {
        println!("  {s}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn extracts_above_min_length() {
        let bytes = b"hello\0world\0";
        let s = extract_strings(bytes, 4);
        assert_eq!(s, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn filters_below_min_length() {
        let bytes = b"hi\0there\0";
        let s = extract_strings(bytes, 4);
        assert_eq!(s, vec!["there".to_string()]);
    }

    #[test]
    fn nul_terminates_string() {
        // String terminated by NUL — what came before, in run, is included.
        let bytes = b"hello\0extra";
        let s = extract_strings(bytes, 4);
        assert_eq!(s, vec!["hello".to_string(), "extra".to_string()]);
    }

    #[test]
    fn nonascii_byte_terminates_string() {
        let bytes = &[b'a', b'b', b'c', b'd', 0xff, b'x', b'y', b'z'];
        let s = extract_strings(bytes, 3);
        assert_eq!(s, vec!["abcd".to_string(), "xyz".to_string()]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(extract_strings(&[], 4).is_empty());
    }

    #[test]
    fn min_len_zero_extracts_everything() {
        // Pathological min_len=0 yields runs of every length (including 0).
        // The 0-length runs are what we get between consecutive non-printable bytes.
        let bytes = b"a\0b";
        let s = extract_strings(bytes, 1);
        assert_eq!(s, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn trailing_run_emitted_if_long_enough() {
        // No NUL terminator at end — run still flushed.
        let bytes = b"trailing";
        let s = extract_strings(bytes, 4);
        assert_eq!(s, vec!["trailing".to_string()]);
    }
}
