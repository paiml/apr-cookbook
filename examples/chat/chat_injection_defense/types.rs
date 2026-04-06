#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;

/// Known dangerous token patterns that could break chat template structure.
pub const DANGEROUS_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|assistant|>",
    "<|system|>",
    "<|user|>",
    "<|end|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "### Instruction:",
    "### Response:",
    "### Input:",
];

/// Patterns that indicate instruction override attempts.
pub const OVERRIDE_PATTERNS: &[&str] = &[
    "ignore previous",
    "ignore all previous",
    "ignore above",
    "disregard previous",
    "disregard all previous",
    "forget previous",
    "forget your instructions",
    "new instructions:",
    "override instructions",
    "system prompt:",
    "you are now",
    "act as if",
    "pretend you are",
    "from now on",
    "instead, do",
];

/// Result of an injection scan.
#[derive(Debug, Clone)]
pub struct InjectionReport {
    pub is_suspicious: bool,
    pub findings: Vec<String>,
}

impl InjectionReport {
    pub fn clean() -> Self {
        Self {
            is_suspicious: false,
            findings: Vec::new(),
        }
    }

    pub fn with_finding(mut self, finding: String) -> Self {
        self.is_suspicious = true;
        self.findings.push(finding);
        self
    }
}

// Sanitize user input by escaping dangerous chat template tokens.
//
// Replaces angle brackets in known token patterns with escaped versions
/// to prevent them from being interpreted as template delimiters.
pub fn sanitize_content(input: &str) -> String {
    let mut result = input.to_string();

    // Escape known dangerous tokens
    for token in DANGEROUS_TOKENS {
        if result.contains(token) {
            let escaped = token
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('[', "&#91;")
                .replace(']', "&#93;");
            result = result.replace(token, &escaped);
        }
    }

    // Remove zero-width characters that could hide payloads
    result = result.replace(
        ['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}', '\u{2060}'],
        "",
    );

    result
}

// Detect potential prompt injection in user input.
//
// Checks for:
// 1. Role-spoofing tokens (e.g., `<|im_start|>system`)
// 2. Instruction override phrases
// 3. Template delimiter injection
/// 4. Suspicious Unicode patterns
pub fn contains_injection(input: &str) -> bool {
    let report = scan_for_injection(input);
    report.is_suspicious
}

/// Perform a detailed injection scan, returning specific findings.
pub fn scan_for_injection(input: &str) -> InjectionReport {
    let mut report = InjectionReport::clean();
    let lower = input.to_lowercase();

    // Check for dangerous token patterns
    for token in DANGEROUS_TOKENS {
        if input.contains(token) {
            report = report.with_finding(format!("Dangerous token found: {token}"));
        }
    }

    // Check for instruction override patterns (case-insensitive)
    for pattern in OVERRIDE_PATTERNS {
        if lower.contains(pattern) {
            report = report.with_finding(format!("Override pattern found: {pattern}"));
        }
    }

    // Check for role-spoofing in angle bracket patterns
    let role_spoof_patterns = ["<|im_start|>system", "<|im_start|>assistant", "<|system|>"];
    for pattern in &role_spoof_patterns {
        if input.contains(pattern) {
            report = report.with_finding(format!("Role spoofing attempt: {pattern}"));
        }
    }

    // Check for zero-width characters (potential payload hiding)
    let zero_width_chars = ['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}', '\u{2060}'];
    for c in &zero_width_chars {
        if input.contains(*c) {
            report =
                report.with_finding(format!("Zero-width character U+{:04X} detected", *c as u32));
        }
    }

    // Check for base64-encoded suspicious content
    if looks_like_base64_payload(&lower) {
        report = report.with_finding("Possible base64-encoded payload detected".to_string());
    }

    report
}

// Heuristic check for base64-encoded payloads.
//
/// Looks for long sequences of base64 characters that could hide instructions.
pub fn looks_like_base64_payload(input: &str) -> bool {
    // Find runs of base64 characters longer than 40 chars
    let mut run_length = 0;
    for c in input.chars() {
        if c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' {
            run_length += 1;
            if run_length > 40 {
                return true;
            }
        } else {
            run_length = 0;
        }
    }
    false
}

/// Apply all defense layers: detect, sanitize, and report.
pub fn defend_input(input: &str) -> (String, InjectionReport) {
    let report = scan_for_injection(input);
    let sanitized = sanitize_content(input);
    (sanitized, report)
}
