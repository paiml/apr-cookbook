//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefectCategory {
    MemorySafety,
    ConcurrencyBugs,
    LogicErrors,
    ApiMisuse,
    ResourceLeaks,
    TypeErrors,
    ConfigurationErrors,
    SecurityVulnerabilities,
    PerformanceIssues,
    IntegrationFailures,
    CodeSmells,
    Complexity,
    DeadCode,
    OwnershipBorrow,
    NullPointerRisk,
    InjectionRisk,
    AuthenticationIssues,
    ErrorHandling,
}

impl DefectCategory {
    #[must_use]
    pub fn severity(&self) -> u8 {
        match self {
            Self::SecurityVulnerabilities
            | Self::InjectionRisk
            | Self::AuthenticationIssues
            | Self::MemorySafety
            | Self::ConcurrencyBugs
            | Self::NullPointerRisk => 4,
            Self::ResourceLeaks
            | Self::OwnershipBorrow
            | Self::ErrorHandling
            | Self::TypeErrors
            | Self::ApiMisuse
            | Self::LogicErrors => 3,
            Self::ConfigurationErrors
            | Self::IntegrationFailures
            | Self::PerformanceIssues
            | Self::Complexity => 2,
            Self::CodeSmells | Self::DeadCode => 1,
        }
    }
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::MemorySafety => "Memory Safety",
            Self::ConcurrencyBugs => "Concurrency Bug",
            Self::LogicErrors => "Logic Error",
            Self::ApiMisuse => "API Misuse",
            Self::ResourceLeaks => "Resource Leak",
            Self::TypeErrors => "Type Error",
            Self::ConfigurationErrors => "Configuration Error",
            Self::SecurityVulnerabilities => "Security Vulnerability",
            Self::PerformanceIssues => "Performance Issue",
            Self::IntegrationFailures => "Integration Failure",
            Self::CodeSmells => "Code Smell",
            Self::Complexity => "High Complexity",
            Self::DeadCode => "Dead Code",
            Self::OwnershipBorrow => "Ownership/Borrow Issue",
            Self::NullPointerRisk => "Null Pointer Risk",
            Self::InjectionRisk => "Injection Risk",
            Self::AuthenticationIssues => "Authentication Issue",
            Self::ErrorHandling => "Error Handling Issue",
        }
    }
    #[must_use]
    pub fn severity_label(&self) -> &'static str {
        match self.severity() {
            4 => "CRITICAL",
            3 => "ERROR",
            2 => "WARNING",
            _ => "INFO",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectionPattern {
    pub name: &'static str,
    pub category: DefectCategory,
    pub keywords: &'static [&'static str],
    pub confidence: f32,
    pub languages: &'static [Language],
    pub suggestion: &'static str,
    pub explanation: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Cpp,
    C,
    Unknown,
}

impl Language {
    #[must_use]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    pub fn detect(code: &str, filename: Option<&str>) -> Self {
        if let Some(name) = filename {
            if name.ends_with(".rs") {
                return Self::Rust;
            }
            if name.ends_with(".py") {
                return Self::Python;
            }
            if name.ends_with(".js") {
                return Self::JavaScript;
            }
            if name.ends_with(".ts") {
                return Self::TypeScript;
            }
            if name.ends_with(".go") {
                return Self::Go;
            }
            if name.ends_with(".java") {
                return Self::Java;
            }
            if name.ends_with(".cpp") || name.ends_with(".cc") || name.ends_with(".cxx") {
                return Self::Cpp;
            }
            if name.ends_with(".c") || name.ends_with(".h") {
                return Self::C;
            }
        }
        if code.contains("fn ") && code.contains("let ") && code.contains("->") {
            return Self::Rust;
        }
        if code.contains("def ") && code.contains(':') && !code.contains('{') {
            return Self::Python;
        }
        if code.contains("func ") && code.contains("package ") {
            return Self::Go;
        }
        if code.contains("function ") || code.contains("const ") || code.contains("=>") {
            if code.contains(": string") || code.contains(": number") {
                return Self::TypeScript;
            }
            return Self::JavaScript;
        }
        if code.contains("public class") || code.contains("private void") {
            return Self::Java;
        }
        if code.contains("#include") || code.contains("std::") {
            return Self::Cpp;
        }
        Self::Unknown
    }
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::JavaScript => "JavaScript",
            Self::TypeScript => "TypeScript",
            Self::Go => "Go",
            Self::Java => "Java",
            Self::Cpp => "C++",
            Self::C => "C",
            Self::Unknown => "Unknown",
        }
    }
}

// PATTERNS const moved to helpers.rs

#[derive(Debug, Clone)]
pub struct DefectFinding {
    pub line: usize,
    pub column: usize,
    pub pattern_name: &'static str,
    pub category: DefectCategory,
    pub confidence: f32,
    pub suspiciousness: f32,
    pub snippet: String,
    pub suggestion: &'static str,
    pub explanation: &'static str,
}

impl DefectFinding {
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "[{}] {}:{}: {} (conf: {:.0}%, susp: {:.0}%)\n  -> {}\n  Snippet: `{}`",
            self.category.severity_label(),
            self.line,
            self.column,
            self.pattern_name,
            self.confidence * 100.0,
            self.suspiciousness * 100.0,
            self.suggestion,
            self.snippet.chars().take(60).collect::<String>()
        )
    }
}

#[derive(Debug, Clone)]
pub struct OracleConfig {
    pub min_confidence: f32,
    pub max_findings: usize,
    pub include_info: bool,
    pub analyze_complexity: bool,
}
impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.4,
            max_findings: 50,
            include_info: false,
            analyze_complexity: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CodeMetrics {
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub max_nesting: usize,
    pub cyclomatic_complexity: usize,
    pub function_count: usize,
}

#[derive(Debug, Clone)]
pub struct OracleReport {
    pub language: Language,
    pub findings: Vec<DefectFinding>,
    pub metrics: CodeMetrics,
    pub health_score: u8,
    pub summary: HashMap<DefectCategory, usize>,
}

impl OracleReport {
    #[must_use]
    pub fn critical_findings(&self) -> Vec<&DefectFinding> {
        self.findings
            .iter()
            .filter(|f| f.category.severity() == 4)
            .collect()
    }
    #[must_use]
    pub fn format(&self) -> String {
        let mut out = format!(
            "# Code Defect Oracle Report\n\n## Language: {}\n\n\
            ## Metrics\n- Lines: {} ({} code, {} comments)\n- Functions: ~{}\n\
            - Max nesting: {}\n- Cyclomatic complexity: ~{}\n\n## Health Score: {}/100\n\n",
            self.language.name(),
            self.metrics.total_lines,
            self.metrics.code_lines,
            self.metrics.comment_lines,
            self.metrics.function_count,
            self.metrics.max_nesting,
            self.metrics.cyclomatic_complexity,
            self.health_score
        );
        if self.findings.is_empty() {
            out.push_str("No defects detected!\n");
        } else {
            out.push_str(&format!("## Findings ({} total)\n\n", self.findings.len()));
            for f in &self.findings {
                out.push_str(&f.format());
                out.push_str("\n\n");
            }
        }
        if !self.summary.is_empty() {
            out.push_str("## Summary by Category\n");
            for (cat, count) in &self.summary {
                out.push_str(&format!("- {}: {}\n", cat.name(), count));
            }
        }
        out
    }
}

#[derive(Debug)]
pub struct CodeOracle {
    pub config: OracleConfig,
}

impl CodeOracle {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OracleConfig::default(),
        }
    }
    #[must_use]
    pub fn with_config(config: OracleConfig) -> Self {
        Self { config }
    }
}

impl Default for CodeOracle {
    fn default() -> Self {
        Self::new()
    }
}
