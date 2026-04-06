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

pub const PATTERNS: &[DetectionPattern] = &[
    DetectionPattern {
        name: "SQL Injection Risk",
        category: DefectCategory::InjectionRisk,
        keywords: &[
            "format!(\"SELECT",
            "format!(\"INSERT",
            "f\"SELECT",
            "f\"INSERT",
            "+ \"SELECT",
            "+ 'SELECT",
            ".format(\"SELECT",
        ],
        confidence: 0.90,
        languages: &[Language::Rust, Language::Python, Language::JavaScript],
        suggestion: "Use parameterized queries",
        explanation: "String interpolation in SQL allows injection",
    },
    DetectionPattern {
        name: "XSS Risk",
        category: DefectCategory::InjectionRisk,
        keywords: &[
            "innerHTML =",
            "innerHTML=",
            "dangerouslySetInnerHTML",
            "document.write(",
            "eval(",
            "v-html=",
        ],
        confidence: 0.85,
        languages: &[Language::JavaScript, Language::TypeScript],
        suggestion: "Use textContent or sanitize HTML",
        explanation: "Unsanitized HTML enables XSS",
    },
    DetectionPattern {
        name: "Command Injection",
        category: DefectCategory::InjectionRisk,
        keywords: &[
            "os.system(",
            "subprocess.call(f\"",
            "subprocess.run(f\"",
            "shell=True",
            "Command::new(&user",
        ],
        confidence: 0.90,
        languages: &[Language::Python, Language::Rust],
        suggestion: "Avoid shell=True and sanitize arguments",
        explanation: "Shell commands with user input enable injection",
    },
    DetectionPattern {
        name: "Hardcoded Credentials",
        category: DefectCategory::AuthenticationIssues,
        keywords: &[
            "password = \"",
            "password=\"",
            "password='",
            "api_key = \"",
            "api_key=\"",
            "secret = \"",
            "AWS_SECRET",
        ],
        confidence: 0.85,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
            Language::Java,
        ],
        suggestion: "Use environment variables or secrets manager",
        explanation: "Hardcoded credentials extractable from source",
    },
    DetectionPattern {
        name: "Buffer Overflow Risk",
        category: DefectCategory::MemorySafety,
        keywords: &[
            "strcpy(", "strcat(", "sprintf(", "gets(", "memcpy(", "unsafe {",
        ],
        confidence: 0.75,
        languages: &[Language::C, Language::Cpp, Language::Rust],
        suggestion: "Use bounds-checked alternatives",
        explanation: "Unchecked buffer ops can overwrite memory",
    },
    DetectionPattern {
        name: "Null Pointer Dereference",
        category: DefectCategory::NullPointerRisk,
        keywords: &[
            ".unwrap()",
            ".expect(",
            "null.",
            "nullptr->",
            "== null",
            "=== null",
        ],
        confidence: 0.65,
        languages: &[
            Language::Rust,
            Language::Java,
            Language::Cpp,
            Language::JavaScript,
            Language::Python,
        ],
        suggestion: "Handle None/null explicitly",
        explanation: "Dereferencing null causes crashes",
    },
    DetectionPattern {
        name: "Data Race Risk",
        category: DefectCategory::ConcurrencyBugs,
        keywords: &[
            "static mut",
            "Arc<Mutex",
            "thread::spawn",
            "go func",
            "pthread_",
            "volatile",
        ],
        confidence: 0.60,
        languages: &[Language::Rust, Language::Go, Language::Java, Language::Cpp],
        suggestion: "Ensure proper synchronization",
        explanation: "Shared mutable state without sync causes races",
    },
    DetectionPattern {
        name: "Ignored Error",
        category: DefectCategory::ErrorHandling,
        keywords: &[
            "let _ =",
            "_ =",
            "catch {}",
            "catch (e) {}",
            "except:",
            "except Exception:",
            "// TODO",
        ],
        confidence: 0.70,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Java,
        ],
        suggestion: "Handle errors explicitly",
        explanation: "Silently ignoring errors hides bugs",
    },
    DetectionPattern {
        name: "Panic in Library",
        category: DefectCategory::ErrorHandling,
        keywords: &[
            "panic!(",
            "unreachable!()",
            "todo!()",
            "unimplemented!()",
            ".unwrap()",
            ".expect(",
        ],
        confidence: 0.65,
        languages: &[Language::Rust],
        suggestion: "Return Result<T, E> instead",
        explanation: "Panics crash the application",
    },
    DetectionPattern {
        name: "Floating Point Comparison",
        category: DefectCategory::LogicErrors,
        keywords: &["== 0.0", "!= 0.0", "f32 ==", "f64 ==", "float =="],
        confidence: 0.80,
        languages: &[Language::Rust, Language::Python, Language::JavaScript],
        suggestion: "Use epsilon comparison",
        explanation: "Direct float equality fails due to precision",
    },
    DetectionPattern {
        name: "Mutable Default Argument",
        category: DefectCategory::LogicErrors,
        keywords: &["def ", "=[]", "={}", "=list()", "=dict()"],
        confidence: 0.85,
        languages: &[Language::Python],
        suggestion: "Use None as default",
        explanation: "Mutable defaults shared between calls",
    },
    DetectionPattern {
        name: "Prototype Pollution Risk",
        category: DefectCategory::SecurityVulnerabilities,
        keywords: &[
            "__proto__",
            "Object.assign(",
            "_.merge(",
            "constructor.prototype",
        ],
        confidence: 0.80,
        languages: &[Language::JavaScript],
        suggestion: "Validate object keys",
        explanation: "Modifying prototypes affects all objects",
    },
    DetectionPattern {
        name: "Long Function",
        category: DefectCategory::Complexity,
        keywords: &[],
        confidence: 0.60,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
        ],
        suggestion: "Split into smaller functions",
        explanation: "Long functions are harder to maintain",
    },
    DetectionPattern {
        name: "Deep Nesting",
        category: DefectCategory::Complexity,
        keywords: &[],
        confidence: 0.65,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
        ],
        suggestion: "Use early returns",
        explanation: "Deep nesting is error-prone",
    },
];

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
    config: OracleConfig,
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

    #[must_use]
    pub fn analyze(&self, code: &str, filename: Option<&str>) -> OracleReport {
        let language = Language::detect(code, filename);
        let metrics = self.compute_metrics(code);
        let mut findings = self.find_defects(code, language);
        self.calculate_suspiciousness(&mut findings, &metrics);
        findings.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if findings.len() > self.config.max_findings {
            findings.truncate(self.config.max_findings);
        }
        let mut summary = HashMap::new();
        for f in &findings {
            *summary.entry(f.category).or_insert(0) += 1;
        }
        let health_score = self.calculate_health_score(&findings, &metrics);
        OracleReport {
            language,
            findings,
            metrics,
            health_score,
            summary,
        }
    }

    pub fn find_defects(&self, code: &str, language: Language) -> Vec<DefectFinding> {
        let mut findings = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        for (line_idx, line) in lines.iter().enumerate() {
            for pattern in PATTERNS {
                if !pattern.languages.contains(&language) && language != Language::Unknown {
                    continue;
                }
                if pattern.confidence < self.config.min_confidence {
                    continue;
                }
                if !self.config.include_info && pattern.category.severity() == 1 {
                    continue;
                }
                for keyword in pattern.keywords {
                    if let Some(col) = line.find(keyword) {
                        findings.push(DefectFinding {
                            line: line_idx + 1,
                            column: col + 1,
                            pattern_name: pattern.name,
                            category: pattern.category,
                            confidence: pattern.confidence,
                            suspiciousness: 0.0,
                            snippet: line.trim().to_string(),
                            suggestion: pattern.suggestion,
                            explanation: pattern.explanation,
                        });
                        break;
                    }
                }
            }
        }
        if self.config.analyze_complexity {
            self.check_complexity(code, &mut findings);
        }
        findings
    }

    pub fn check_complexity(&self, code: &str, findings: &mut Vec<DefectFinding>) {
        let lines: Vec<&str> = code.lines().collect();
        let (mut depth, mut max_depth, mut max_line): (usize, usize, usize) = (0, 0, 0);
        let (mut in_fn, mut fn_start, mut fn_lines) = (false, 0usize, 0usize);
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            let opens = line.chars().filter(|&c| c == '{').count();
            let closes = line.chars().filter(|&c| c == '}').count();
            depth = depth.saturating_add(opens);
            if depth > max_depth {
                max_depth = depth;
                max_line = idx + 1;
            }
            depth = depth.saturating_sub(closes);
            let indent_depth = (line.len() - line.trim_start().len()) / 4;
            if indent_depth > max_depth {
                max_depth = indent_depth;
                max_line = idx + 1;
            }
            if trimmed.starts_with("fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("func ")
            {
                if in_fn && fn_lines > 50 {
                    findings.push(DefectFinding {
                        line: fn_start,
                        column: 1,
                        pattern_name: "Long Function",
                        category: DefectCategory::Complexity,
                        confidence: 0.60,
                        suspiciousness: 0.0,
                        snippet: lines.get(fn_start - 1).unwrap_or(&"").trim().to_string(),
                        suggestion: "Split into smaller functions",
                        explanation: "Long functions are harder to maintain",
                    });
                }
                in_fn = true;
                fn_start = idx + 1;
                fn_lines = 0;
            }
            if in_fn {
                fn_lines += 1;
            }
        }
        if in_fn && fn_lines > 50 {
            findings.push(DefectFinding {
                line: fn_start,
                column: 1,
                pattern_name: "Long Function",
                category: DefectCategory::Complexity,
                confidence: 0.60,
                suspiciousness: 0.0,
                snippet: lines.get(fn_start - 1).unwrap_or(&"").trim().to_string(),
                suggestion: "Split into smaller functions",
                explanation: "Long functions are harder to maintain",
            });
        }
        if max_depth > 4 {
            findings.push(DefectFinding {
                line: max_line,
                column: 1,
                pattern_name: "Deep Nesting",
                category: DefectCategory::Complexity,
                confidence: 0.65,
                suspiciousness: 0.0,
                snippet: lines.get(max_line - 1).unwrap_or(&"").trim().to_string(),
                suggestion: "Use early returns",
                explanation: "Deep nesting is error-prone",
            });
        }
    }

    pub fn compute_metrics(&self, code: &str) -> CodeMetrics {
        let lines: Vec<&str> = code.lines().collect();
        let (mut code_lines, mut comment_lines, mut max_nesting, mut cur_nesting) =
            (0usize, 0usize, 0usize, 0usize);
        let (mut fn_count, mut branch_count) = (0usize, 0usize);
        for line in &lines {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with("//")
                || t.starts_with('#')
                || t.starts_with("/*")
                || t.starts_with('*')
            {
                comment_lines += 1;
            } else {
                code_lines += 1;
            }
            if t.starts_with("fn ")
                || t.starts_with("def ")
                || t.starts_with("function ")
                || t.starts_with("func ")
                || t.contains("=> {")
            {
                fn_count += 1;
            }
            if t.starts_with("if ")
                || t.starts_with("else if")
                || t.starts_with("elif ")
                || t.starts_with("while ")
                || t.starts_with("for ")
                || t.starts_with("match ")
                || t.contains("&&")
                || t.contains("||")
            {
                branch_count += 1;
            }
            cur_nesting = cur_nesting.saturating_add(line.chars().filter(|&c| c == '{').count());
            max_nesting = max_nesting.max(cur_nesting);
            cur_nesting = cur_nesting.saturating_sub(line.chars().filter(|&c| c == '}').count());
            max_nesting = max_nesting.max((line.len() - line.trim_start().len()) / 4);
        }
        CodeMetrics {
            total_lines: lines.len(),
            code_lines,
            comment_lines,
            max_nesting,
            cyclomatic_complexity: branch_count + 1,
            function_count: fn_count,
        }
    }

    pub fn calculate_suspiciousness(&self, findings: &mut [DefectFinding], metrics: &CodeMetrics) {
        let cf = if metrics.cyclomatic_complexity > 20 {
            1.2
        } else if metrics.cyclomatic_complexity > 10 {
            1.1
        } else {
            1.0
        };
        for f in findings {
            f.suspiciousness =
                ((f.confidence * 0.6 + f32::from(f.category.severity()) / 4.0 * 0.4) * cf).min(1.0);
        }
    }

    pub fn calculate_health_score(&self, findings: &[DefectFinding], metrics: &CodeMetrics) -> u8 {
        let mut score = 100i32;
        for f in findings {
            match f.category.severity() {
                4 => score -= 15,
                3 => score -= 8,
                2 => score -= 3,
                _ => score -= 1,
            }
        }
        if metrics.cyclomatic_complexity > 30 {
            score -= 10;
        } else if metrics.cyclomatic_complexity > 20 {
            score -= 5;
        }
        if metrics.max_nesting > 6 {
            score -= 10;
        } else if metrics.max_nesting > 4 {
            score -= 5;
        }
        score.clamp(0, 100) as u8
    }
}

impl Default for CodeOracle {
    fn default() -> Self {
        Self::new()
    }
}
