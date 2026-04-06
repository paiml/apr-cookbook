#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "PASS"),
            CheckStatus::Fail => write!(f, "FAIL"),
            CheckStatus::Warn => write!(f, "WARN"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: u32,
    pub failed: u32,
    pub warnings: u32,
    pub checks: Vec<Check>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            warnings: 0,
            checks: Vec::new(),
        }
    }

    pub fn add(&mut self, name: &str, status: CheckStatus, detail: &str) {
        match status {
            CheckStatus::Pass => self.passed += 1,
            CheckStatus::Fail => self.failed += 1,
            CheckStatus::Warn => self.warnings += 1,
        }
        self.checks.push(Check {
            name: name.to_string(),
            status,
            detail: detail.to_string(),
        });
    }

    pub fn score(&self) -> u32 {
        let total = self.passed + self.failed + self.warnings;
        if total == 0 {
            return 0;
        }
        // Each pass = full points, warn = half, fail = 0
        let earned = self.passed * 100 + self.warnings * 50;
        earned / total
    }

    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

pub fn validate_model(bytes: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::new();
    check_magic_bytes(bytes, &mut result);
    check_minimum_size(bytes, &mut result);
    check_version(bytes, &mut result);
    check_metadata_present(bytes, &mut result);
    check_tensors_nonempty(bytes, &mut result);
    check_no_nan(bytes, &mut result);
    check_no_inf(bytes, &mut result);
    check_compression_valid(bytes, &mut result);
    check_size_consistent(bytes, &mut result);
    check_checksum(bytes, &mut result);
    result
}

pub fn check_magic_bytes(bytes: &[u8], result: &mut ValidationResult) {
    let (status, detail) = if bytes.len() >= 4 && &bytes[0..4] == b"APR2" {
        (CheckStatus::Pass, "Valid APR2 magic bytes".into())
    } else if bytes.len() >= 4 {
        (
            CheckStatus::Fail,
            format!("Invalid magic: {:?}", &bytes[0..4]),
        )
    } else {
        (CheckStatus::Fail, "File too small for magic bytes".into())
    };
    result.add("magic_bytes", status, &detail);
}

pub fn check_minimum_size(bytes: &[u8], result: &mut ValidationResult) {
    let (status, op) = if bytes.len() >= 64 {
        (CheckStatus::Pass, ">=")
    } else {
        (CheckStatus::Fail, "<")
    };
    result.add(
        "minimum_size",
        status,
        &format!("{} bytes ({op} 64 byte minimum)", bytes.len()),
    );
}

pub fn check_version(bytes: &[u8], result: &mut ValidationResult) {
    if bytes.len() <= 4 {
        result.add(
            "format_version",
            CheckStatus::Fail,
            "File too small for version byte",
        );
        return;
    }
    let version = bytes[4];
    let (status, detail) = match version {
        0 | 2 => (CheckStatus::Pass, format!("Version byte: {version}")),
        1 => (
            CheckStatus::Warn,
            "Legacy v1 format detected; consider upgrading".into(),
        ),
        _ => (CheckStatus::Fail, format!("Unknown version: {version}")),
    };
    result.add("format_version", status, &detail);
}

pub fn check_metadata_present(bytes: &[u8], result: &mut ValidationResult) {
    let has_name = bytes.windows(5).any(|w| w == b"name=");
    let has_desc = bytes.windows(12).any(|w| w == b"description=");
    let (status, detail) = if has_name {
        (CheckStatus::Pass, "Model name found in metadata")
    } else if has_desc {
        (CheckStatus::Warn, "Name missing but description present")
    } else {
        (CheckStatus::Warn, "No standard metadata fields found")
    };
    result.add("metadata_present", status, detail);
}

pub fn check_tensors_nonempty(bytes: &[u8], result: &mut ValidationResult) {
    let header_size = 64;
    let (status, detail) = if bytes.len() > header_size + 16 {
        let payload_size = bytes.len() - header_size;
        (
            CheckStatus::Pass,
            format!("Payload size: {payload_size} bytes"),
        )
    } else if bytes.len() > header_size {
        (
            CheckStatus::Warn,
            "Payload very small; model may be trivial".into(),
        )
    } else {
        (CheckStatus::Fail, "No tensor payload detected".into())
    };
    result.add("tensors_nonempty", status, &detail);
}

pub fn check_no_nan(bytes: &[u8], result: &mut ValidationResult) {
    let n = count_special_values(bytes, f32::is_nan);
    let (s, d) = if n == 0 {
        (CheckStatus::Pass, "No NaN values detected".into())
    } else {
        (
            CheckStatus::Fail,
            format!("{n} NaN value(s) detected in tensor data"),
        )
    };
    result.add("no_nan", s, &d);
}

pub fn check_no_inf(bytes: &[u8], result: &mut ValidationResult) {
    let n = count_special_values(bytes, f32::is_infinite);
    let (s, d) = if n == 0 {
        (CheckStatus::Pass, "No Inf values detected".into())
    } else {
        (
            CheckStatus::Warn,
            format!("{n} Inf value(s) detected in tensor data"),
        )
    };
    result.add("no_inf", s, &d);
}

pub fn check_compression_valid(bytes: &[u8], result: &mut ValidationResult) {
    let lz4_magic: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
    let zstd_magic: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
    let has_lz4 = bytes.windows(4).any(|w| w == lz4_magic);
    let has_zstd = bytes.windows(4).any(|w| w == zstd_magic);
    let (status, detail) = if has_lz4 && has_zstd {
        (
            CheckStatus::Warn,
            "Multiple compression formats detected".into(),
        )
    } else if has_lz4 || has_zstd || bytes.len() < 256 {
        let m = if has_lz4 {
            "LZ4"
        } else if has_zstd {
            "Zstd"
        } else {
            "None/uncompressed"
        };
        (CheckStatus::Pass, format!("Compression: {m}"))
    } else {
        (
            CheckStatus::Pass,
            "No compression detected (raw payload)".into(),
        )
    };
    result.add("compression_valid", status, &detail);
}

pub fn check_size_consistent(bytes: &[u8], result: &mut ValidationResult) {
    let (status, detail) = if bytes.len() >= 64 && bytes.len() % 4 == 0 {
        (CheckStatus::Pass, "File size is 4-byte aligned")
    } else if bytes.len() >= 64 {
        (
            CheckStatus::Warn,
            "File size not 4-byte aligned; possible truncation",
        )
    } else {
        (CheckStatus::Fail, "File smaller than minimum header size")
    };
    result.add("size_consistent", status, detail);
}

pub fn check_checksum(bytes: &[u8], result: &mut ValidationResult) {
    let cs = compute_checksum(bytes);
    let (s, d) = if cs > 0 {
        (CheckStatus::Pass, format!("Checksum: 0x{cs:08X}"))
    } else {
        (
            CheckStatus::Warn,
            "Zero checksum; payload may be all zeros".into(),
        )
    };
    result.add("checksum", s, &d);
}

pub fn count_special_values(bytes: &[u8], pred: fn(f32) -> bool) -> usize {
    let start = 64.min(bytes.len());
    bytes[start..]
        .chunks_exact(4)
        .filter(|c| pred(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))))
        .count()
}

pub fn compute_checksum(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV-1a offset basis
    for &byte in bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
    }
    hash
}

pub fn inject_nan_at(bytes: &mut [u8], offset: usize) {
    // IEEE 754 NaN: exponent all 1s, non-zero mantissa
    let nan_bits: u32 = 0x7FC0_0000;
    let nan_bytes = nan_bits.to_le_bytes();
    if offset + 4 <= bytes.len() {
        bytes[offset..offset + 4].copy_from_slice(&nan_bytes);
    }
}

#[cfg(test)]
pub fn inject_inf_at(bytes: &mut [u8], offset: usize) {
    let inf_bits: u32 = 0x7F80_0000;
    let inf_bytes = inf_bits.to_le_bytes();
    if offset + 4 <= bytes.len() {
        bytes[offset..offset + 4].copy_from_slice(&inf_bytes);
    }
}
