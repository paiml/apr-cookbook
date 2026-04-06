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

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Result of a single check stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub name: String,
    pub passed: bool,
    pub skipped: bool,
    pub detail: String,
}

impl StageResult {
    pub fn pass(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            skipped: false,
            detail: detail.to_string(),
        }
    }

    pub fn fail(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            skipped: false,
            detail: detail.to_string(),
        }
    }

    pub fn skip(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            skipped: true,
            detail: detail.to_string(),
        }
    }

    pub fn status_str(&self) -> &str {
        if self.skipped {
            "SKIP"
        } else if self.passed {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

/// Overall verdict for the check report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckVerdict {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for CheckVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckVerdict::Pass => write!(f, "PASS"),
            CheckVerdict::Fail => write!(f, "FAIL"),
            CheckVerdict::Warn => write!(f, "WARN"),
        }
    }
}

/// Full check report across all stages.
#[derive(Debug, Clone)]
pub struct CheckReport {
    pub model_name: String,
    pub stages: Vec<StageResult>,
}

impl CheckReport {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            stages: Vec::with_capacity(10),
        }
    }

    pub fn add(&mut self, result: StageResult) {
        self.stages.push(result);
    }

    pub fn passed_count(&self) -> usize {
        self.stages.iter().filter(|s| s.passed).count()
    }

    pub fn failed_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| !s.passed && !s.skipped)
            .count()
    }

    pub fn skipped_count(&self) -> usize {
        self.stages.iter().filter(|s| s.skipped).count()
    }

    pub fn verdict(&self) -> CheckVerdict {
        if self.failed_count() > 0 {
            CheckVerdict::Fail
        } else if self.skipped_count() > 0 {
            CheckVerdict::Warn
        } else {
            CheckVerdict::Pass
        }
    }
}

// ---------------------------------------------------------------------------
// Known dtype byte values (APR v2 header byte 7)
// ---------------------------------------------------------------------------

pub const KNOWN_DTYPES: [u8; 4] = [0, 1, 2, 3]; // FP32, FP16, Int8, Int4

// ---------------------------------------------------------------------------
// Stage implementations
// ---------------------------------------------------------------------------

// Parse all tensor shapes from the APR v2 tensor index.
//
// Returns `None` if the header is too short; otherwise returns a vector of

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
