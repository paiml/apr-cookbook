//! # apr rosetta inspect — `--hexdump` Header Window
//!
//! `apr rosetta inspect <FILE> --hexdump` shows the first N bytes of the
//! file in classic xxd-style format (offset + 16 bytes hex + ASCII gutter).
//! This recipe builds the formatter as a pure function so a CI pipeline
//! can preview the dump format and assert the contract: lines exactly 16
//! bytes wide except the trailing one, ASCII gutter shows `.` for
//! non-printable bytes.
//!
//! Demonstrates the **ROSETTA-INSPECT.2** recipe for PMAT-098 (apr rosetta inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001 + xxd(1) classic layout
//!
//! Run with: cargo run --example cli_rosetta_inspect_hexdump_window
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BYTES_PER_LINE: usize = 16;

pub fn hexdump(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (line_idx, chunk) in bytes.chunks(BYTES_PER_LINE).enumerate() {
        let offset = line_idx * BYTES_PER_LINE;
        let _ = write!(out, "{offset:08x}  ");
        for i in 0..BYTES_PER_LINE {
            if i < chunk.len() {
                let _ = write!(out, "{:02x} ", chunk[i]);
            } else {
                out.push_str("   ");
            }
            if i == 7 {
                out.push(' ');
            }
        }
        out.push_str(" |");
        for &b in chunk {
            out.push(if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            });
        }
        out.push('|');
        out.push('\n');
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_inspect_hexdump_window")?;

    let mut sample = b"APR2".to_vec();
    sample.extend_from_slice(&[0u8; 12]);
    sample.extend_from_slice(b"Hello, world!\n");
    sample.extend_from_slice(&(0..18u8).collect::<Vec<_>>());

    println!("{}", hexdump(&sample));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hexdump_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_yields_empty_dump() {
        assert_eq!(hexdump(&[]), "");
    }

    #[test]
    fn single_byte_dump_pads_to_16_columns() {
        let dump = hexdump(&[0x41]); // 'A'
                                     // Single line: offset + hex (one byte + 15 padding triples) + gutter.
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 1);
        // Verify line ends with the ASCII gutter `|A|`.
        assert!(lines[0].ends_with("|A|"));
    }

    #[test]
    fn full_16_bytes_yields_one_line() {
        let bytes: Vec<u8> = (0..16).collect();
        let dump = hexdump(&bytes);
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn seventeen_bytes_yields_two_lines() {
        let bytes: Vec<u8> = (0..17).collect();
        let dump = hexdump(&bytes);
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 2);
        // Second line offset must be 0x00000010.
        assert!(lines[1].starts_with("00000010"));
    }

    #[test]
    fn nonprintable_bytes_render_as_dots_in_gutter() {
        let dump = hexdump(&[0x00, 0x01, 0xff]);
        // ASCII gutter must show `...`.
        assert!(dump.contains("|...|"));
    }

    #[test]
    fn printable_ascii_renders_in_gutter() {
        let dump = hexdump(b"Hello!");
        // Gutter contains "Hello!".
        assert!(dump.contains("|Hello!|"));
    }

    #[test]
    fn space_treated_as_printable() {
        // Space (0x20) IS printable in standard hexdump output.
        let dump = hexdump(b" ");
        assert!(dump.contains("| |"));
    }
}
