//! # Demo D: Code Defect Oracle
//!
//! A browser-ready code analysis tool that detects defects, security issues,
//! and anti-patterns in code. Designed for WASM deployment where users paste
//! code and receive instant feedback on potential issues.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic defect detection at the source
//! - **Genchi Genbutsu**: Analyze actual code patterns, not abstractions
//! - **Poka-yoke**: Prevent defects before they reach production
//! - **Kaizen**: Continuous improvement through pattern learning
//!
//! ## Features
//!
//! - Multi-language support (Rust, Python, JavaScript, Go)
//! - 18 defect categories with confidence scoring
//! - Tarantula-style fault localization
//! - Actionable fix suggestions
//! - WASM-compatible design (no std dependencies in core)
//!
//! ## Architecture
//!
//! ```text
//! Code Input → Tokenizer → Pattern Matcher → Classifier → Ranker → Report
//!                              ↓
//!                     Rule Engine + ML Features
//! ```

use std::collections::HashMap;

// ============================================================================
// DEFECT CATEGORIES (18 categories from OIP taxonomy)
// ============================================================================

/// Defect category with severity and description
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefectCategory {
    // General categories
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
    // Code quality
    CodeSmells,
    Complexity,
    DeadCode,
    // Language-specific
    OwnershipBorrow,      // Rust
    NullPointerRisk,      // Java/C/C++
    InjectionRisk,        // SQL/XSS
    AuthenticationIssues, // Security
    ErrorHandling,        // All languages
}

impl DefectCategory {
    /// Get severity level (1-4)
    #[must_use]
    pub fn severity(&self) -> u8 {
        match self {
            Self::SecurityVulnerabilities | Self::InjectionRisk | Self::AuthenticationIssues => 4,
            Self::MemorySafety | Self::ConcurrencyBugs | Self::NullPointerRisk => 4,
            Self::ResourceLeaks | Self::OwnershipBorrow | Self::ErrorHandling => 3,
            Self::TypeErrors | Self::ApiMisuse | Self::LogicErrors => 3,
            Self::ConfigurationErrors | Self::IntegrationFailures => 2,
            Self::PerformanceIssues | Self::Complexity => 2,
            Self::CodeSmells | Self::DeadCode => 1,
        }
    }

    /// Get human-readable name
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

    /// Get severity label
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

// ============================================================================
// DETECTION PATTERNS
// ============================================================================

/// A pattern rule for detecting defects
#[derive(Debug, Clone)]
pub struct DetectionPattern {
    /// Pattern name
    pub name: &'static str,
    /// Category of defect
    pub category: DefectCategory,
    /// Keywords to match (any match triggers)
    pub keywords: &'static [&'static str],
    /// Confidence when matched (0.0-1.0)
    pub confidence: f32,
    /// Languages this applies to
    pub languages: &'static [Language],
    /// Suggested fix
    pub suggestion: &'static str,
    /// Explanation
    pub explanation: &'static str,
}

/// Supported programming languages
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
    /// Detect language from file extension or content
    #[must_use]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    pub fn detect(code: &str, filename: Option<&str>) -> Self {
        // Check filename extension first
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

        // Heuristic detection from content
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

    /// Get language name
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

/// All detection patterns (18 categories)
pub const PATTERNS: &[DetectionPattern] = &[
    // === CRITICAL: Security ===
    DetectionPattern {
        name: "SQL Injection Risk",
        category: DefectCategory::InjectionRisk,
        keywords: &[
            "format!(\"SELECT",
            "format!(\"INSERT",
            "format!(\"UPDATE",
            "format!(\"DELETE",
            "f\"SELECT",
            "f\"INSERT",
            "f'SELECT",
            "+ \"SELECT",
            "+ 'SELECT",
            "execute(f\"",
            ".format(\"SELECT",
        ],
        confidence: 0.90,
        languages: &[Language::Rust, Language::Python, Language::JavaScript],
        suggestion: "Use parameterized queries instead of string interpolation",
        explanation: "String interpolation in SQL queries allows attackers to inject malicious SQL",
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
            ".html(user",
            "v-html=",
        ],
        confidence: 0.85,
        languages: &[Language::JavaScript, Language::TypeScript],
        suggestion: "Use textContent or sanitize HTML before insertion",
        explanation:
            "Inserting unsanitized user input as HTML enables cross-site scripting attacks",
    },
    DetectionPattern {
        name: "Command Injection",
        category: DefectCategory::InjectionRisk,
        keywords: &[
            "os.system(",
            "subprocess.call(f\"",
            "subprocess.run(f\"",
            "exec(user",
            "exec(input",
            "shell=True",
            "Command::new(&user",
        ],
        confidence: 0.90,
        languages: &[Language::Python, Language::Rust],
        suggestion: "Avoid shell=True and validate/sanitize all command arguments",
        explanation: "Executing shell commands with user input enables command injection attacks",
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
            "SECRET_KEY = \"",
            "AWS_SECRET",
            "private_key = \"",
        ],
        confidence: 0.85,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
            Language::Java,
        ],
        suggestion: "Use environment variables or a secrets manager",
        explanation: "Hardcoded credentials can be extracted from source code or binaries",
    },
    // === CRITICAL: Memory Safety ===
    DetectionPattern {
        name: "Use After Free Risk",
        category: DefectCategory::MemorySafety,
        keywords: &["free(", "delete ", "drop(", ".take()", "ManuallyDrop"],
        confidence: 0.70,
        languages: &[Language::C, Language::Cpp, Language::Rust],
        suggestion: "Ensure pointer is not used after deallocation",
        explanation: "Using memory after freeing it causes undefined behavior and security issues",
    },
    DetectionPattern {
        name: "Buffer Overflow Risk",
        category: DefectCategory::MemorySafety,
        keywords: &[
            "strcpy(", "strcat(", "sprintf(", "gets(", "[index]", "memcpy(", "unsafe {",
        ],
        confidence: 0.75,
        languages: &[Language::C, Language::Cpp, Language::Rust],
        suggestion: "Use bounds-checked alternatives (strncpy, snprintf) or safe wrappers",
        explanation: "Unchecked buffer operations can overwrite adjacent memory",
    },
    DetectionPattern {
        name: "Null Pointer Dereference",
        category: DefectCategory::NullPointerRisk,
        keywords: &[
            ".unwrap()",
            ".expect(",
            "None =>",
            "null.",
            "nullptr->",
            "== null",
            "=== null",
            "is None",
        ],
        confidence: 0.65,
        languages: &[
            Language::Rust,
            Language::Java,
            Language::Cpp,
            Language::JavaScript,
            Language::Python,
        ],
        suggestion: "Handle the None/null case explicitly or use safe unwrapping",
        explanation: "Dereferencing null/None causes crashes or undefined behavior",
    },
    // === ERROR: Concurrency ===
    DetectionPattern {
        name: "Data Race Risk",
        category: DefectCategory::ConcurrencyBugs,
        keywords: &[
            "static mut",
            "Arc<Mutex",
            "thread::spawn",
            "go func",
            "pthread_",
            "synchronized",
            "volatile",
            "AtomicUsize",
        ],
        confidence: 0.60,
        languages: &[Language::Rust, Language::Go, Language::Java, Language::Cpp],
        suggestion: "Ensure proper synchronization and consider using channels",
        explanation:
            "Concurrent access to shared mutable state without synchronization causes data races",
    },
    DetectionPattern {
        name: "Deadlock Risk",
        category: DefectCategory::ConcurrencyBugs,
        keywords: &[
            ".lock()",
            ".lock().unwrap()",
            "mutex.Lock()",
            "synchronized(",
            "pthread_mutex_lock",
            "ReentrantLock",
        ],
        confidence: 0.55,
        languages: &[Language::Rust, Language::Go, Language::Java, Language::Cpp],
        suggestion: "Use lock ordering or try_lock with timeout",
        explanation: "Multiple locks acquired in different orders can cause deadlocks",
    },
    // === ERROR: Resource Leaks ===
    DetectionPattern {
        name: "Resource Leak",
        category: DefectCategory::ResourceLeaks,
        keywords: &[
            "File::open",
            "open(",
            "fopen(",
            "socket(",
            "connect(",
            "new FileInputStream",
            "createConnection",
        ],
        confidence: 0.50,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::C,
            Language::Java,
            Language::JavaScript,
        ],
        suggestion: "Use RAII, context managers, or try-with-resources",
        explanation: "Opened resources must be closed to prevent leaks",
    },
    // === ERROR: Error Handling ===
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
            "// ignore",
            "// TODO",
            "pass  #",
        ],
        confidence: 0.70,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Java,
        ],
        suggestion: "Handle errors explicitly or propagate with ?",
        explanation: "Silently ignoring errors hides bugs and makes debugging difficult",
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
        suggestion: "Return Result<T, E> instead of panicking in library code",
        explanation: "Panics in libraries crash the entire application unexpectedly",
    },
    // === WARNING: Logic Errors ===
    DetectionPattern {
        name: "Off-by-One Risk",
        category: DefectCategory::LogicErrors,
        keywords: &[
            "< len",
            "<= len",
            "length - 1",
            "len() - 1",
            ".len() -",
            "for i in 0..",
            "range(len(",
        ],
        confidence: 0.45,
        languages: &[Language::Rust, Language::Python, Language::JavaScript],
        suggestion: "Double-check loop bounds and array indices",
        explanation: "Off-by-one errors are among the most common bugs in loops and array access",
    },
    DetectionPattern {
        name: "Floating Point Comparison",
        category: DefectCategory::LogicErrors,
        keywords: &["== 0.0", "!= 0.0", "== 0.", "f32 ==", "f64 ==", "float =="],
        confidence: 0.80,
        languages: &[Language::Rust, Language::Python, Language::JavaScript],
        suggestion: "Use epsilon comparison: (a - b).abs() < epsilon",
        explanation: "Direct floating-point equality comparison fails due to precision issues",
    },
    // === WARNING: Performance ===
    DetectionPattern {
        name: "N+1 Query Pattern",
        category: DefectCategory::PerformanceIssues,
        keywords: &[
            "for item in",
            "for row in",
            ".all()",
            "SELECT * FROM",
            "findAll(",
            ".find(",
        ],
        confidence: 0.40,
        languages: &[Language::Python, Language::JavaScript, Language::Java],
        suggestion: "Use eager loading or batch queries",
        explanation: "Querying in a loop causes N+1 queries, severely impacting performance",
    },
    DetectionPattern {
        name: "String Concatenation in Loop",
        category: DefectCategory::PerformanceIssues,
        keywords: &["+= \"", "+= '", "result += ", "output = output +", "str +="],
        confidence: 0.70,
        languages: &[Language::Python, Language::JavaScript, Language::Java],
        suggestion: "Use StringBuilder, join(), or collect::<String>()",
        explanation: "String concatenation in loops creates many temporary strings",
    },
    // === INFO: Code Smells ===
    DetectionPattern {
        name: "Magic Number",
        category: DefectCategory::CodeSmells,
        keywords: &[
            "== 0", "== 1", "== 2", "== 100", "== 1000", "* 60", "* 24", "* 1024",
        ],
        confidence: 0.35,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Java,
        ],
        suggestion: "Extract magic numbers to named constants",
        explanation: "Magic numbers reduce code readability and maintainability",
    },
    DetectionPattern {
        name: "Long Function",
        category: DefectCategory::Complexity,
        keywords: &[], // Detected by line count
        confidence: 0.60,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
        ],
        suggestion: "Split into smaller, focused functions",
        explanation: "Long functions are harder to understand, test, and maintain",
    },
    DetectionPattern {
        name: "Deep Nesting",
        category: DefectCategory::Complexity,
        keywords: &[], // Detected by indentation analysis
        confidence: 0.65,
        languages: &[
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::Go,
        ],
        suggestion: "Use early returns or extract nested logic",
        explanation: "Deeply nested code is hard to follow and error-prone",
    },
    // === Language-specific: Python ===
    DetectionPattern {
        name: "Mutable Default Argument",
        category: DefectCategory::LogicErrors,
        keywords: &["def ", "=[]", "={}", "=list()", "=dict()", "=set()"],
        confidence: 0.85,
        languages: &[Language::Python],
        suggestion: "Use None as default and create mutable object inside function",
        explanation: "Mutable default arguments are shared between calls, causing subtle bugs",
    },
    DetectionPattern {
        name: "Bare Except",
        category: DefectCategory::ErrorHandling,
        keywords: &["except:", "except Exception:"],
        confidence: 0.75,
        languages: &[Language::Python],
        suggestion: "Catch specific exceptions",
        explanation: "Bare except catches SystemExit and KeyboardInterrupt unexpectedly",
    },
    // === Language-specific: JavaScript ===
    DetectionPattern {
        name: "Loose Equality",
        category: DefectCategory::TypeErrors,
        keywords: &["== null", "== undefined", "!= null", "!= undefined"],
        confidence: 0.70,
        languages: &[Language::JavaScript],
        suggestion: "Use === and !== for strict equality",
        explanation: "Loose equality performs type coercion, leading to unexpected results",
    },
    DetectionPattern {
        name: "Prototype Pollution Risk",
        category: DefectCategory::SecurityVulnerabilities,
        keywords: &[
            "__proto__",
            "Object.assign(",
            "_.merge(",
            "$.extend(",
            "constructor.prototype",
        ],
        confidence: 0.80,
        languages: &[Language::JavaScript],
        suggestion: "Validate object keys and use Object.create(null)",
        explanation: "Modifying prototypes can affect all objects and enable attacks",
    },
    // === Language-specific: Rust ===
    DetectionPattern {
        name: "Unnecessary Clone",
        category: DefectCategory::PerformanceIssues,
        keywords: &[".clone()", ".to_owned()", ".to_string()"],
        confidence: 0.40,
        languages: &[Language::Rust],
        suggestion: "Consider borrowing instead of cloning",
        explanation: "Cloning allocates new memory; borrowing is often sufficient",
    },
    DetectionPattern {
        name: "Missing Error Context",
        category: DefectCategory::ErrorHandling,
        keywords: &[".map_err(|_|", ".map_err(|e| e)", "Err(e) => Err(e)"],
        confidence: 0.55,
        languages: &[Language::Rust],
        suggestion: "Add context with anyhow::Context or thiserror",
        explanation: "Errors without context make debugging difficult",
    },
];

// ============================================================================
// DEFECT FINDING
// ============================================================================

/// A detected defect in code
#[derive(Debug, Clone)]
pub struct DefectFinding {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// The matched pattern
    pub pattern_name: &'static str,
    /// Defect category
    pub category: DefectCategory,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Suspiciousness score (Tarantula-style, 0.0-1.0)
    pub suspiciousness: f32,
    /// The offending code snippet
    pub snippet: String,
    /// Suggested fix
    pub suggestion: &'static str,
    /// Explanation
    pub explanation: &'static str,
}

impl DefectFinding {
    /// Format as a report line
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "[{}] {}:{}: {} (confidence: {:.0}%, suspiciousness: {:.0}%)\n  → {}\n  Snippet: `{}`",
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

// ============================================================================
// CODE ORACLE
// ============================================================================

/// Analysis configuration
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Minimum confidence to report
    pub min_confidence: f32,
    /// Maximum findings to return
    pub max_findings: usize,
    /// Include info-level findings
    pub include_info: bool,
    /// Calculate complexity metrics
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

/// Code analysis metrics
#[derive(Debug, Clone, Default)]
pub struct CodeMetrics {
    /// Total lines
    pub total_lines: usize,
    /// Non-empty lines
    pub code_lines: usize,
    /// Comment lines
    pub comment_lines: usize,
    /// Maximum nesting depth
    pub max_nesting: usize,
    /// Cyclomatic complexity estimate
    pub cyclomatic_complexity: usize,
    /// Number of functions
    pub function_count: usize,
}

/// Analysis report
#[derive(Debug, Clone)]
pub struct OracleReport {
    /// Detected language
    pub language: Language,
    /// All findings sorted by suspiciousness
    pub findings: Vec<DefectFinding>,
    /// Code metrics
    pub metrics: CodeMetrics,
    /// Overall health score (0-100)
    pub health_score: u8,
    /// Summary by category
    pub summary: HashMap<DefectCategory, usize>,
}

impl OracleReport {
    /// Get critical findings only
    #[must_use]
    pub fn critical_findings(&self) -> Vec<&DefectFinding> {
        self.findings
            .iter()
            .filter(|f| f.category.severity() == 4)
            .collect()
    }

    /// Format as text report
    #[must_use]
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "# Code Defect Oracle Report\n\n## Language: {}\n\n",
            self.language.name()
        ));

        output.push_str(&format!(
            "## Metrics\n- Lines: {} ({} code, {} comments)\n- Functions: ~{}\n- Max nesting: {}\n- Cyclomatic complexity: ~{}\n\n",
            self.metrics.total_lines,
            self.metrics.code_lines,
            self.metrics.comment_lines,
            self.metrics.function_count,
            self.metrics.max_nesting,
            self.metrics.cyclomatic_complexity
        ));

        output.push_str(&format!("## Health Score: {}/100\n\n", self.health_score));

        if self.findings.is_empty() {
            output.push_str("✅ No defects detected!\n");
        } else {
            output.push_str(&format!("## Findings ({} total)\n\n", self.findings.len()));

            for finding in &self.findings {
                output.push_str(&finding.format());
                output.push_str("\n\n");
            }
        }

        // Summary
        if !self.summary.is_empty() {
            output.push_str("## Summary by Category\n");
            for (cat, count) in &self.summary {
                output.push_str(&format!("- {}: {}\n", cat.name(), count));
            }
        }

        output
    }
}

/// The Code Defect Oracle
#[derive(Debug)]
pub struct CodeOracle {
    config: OracleConfig,
}

impl CodeOracle {
    /// Create a new oracle with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OracleConfig::default(),
        }
    }

    /// Create with custom config
    #[must_use]
    pub fn with_config(config: OracleConfig) -> Self {
        Self { config }
    }

    /// Analyze code and return report
    #[must_use]
    pub fn analyze(&self, code: &str, filename: Option<&str>) -> OracleReport {
        let language = Language::detect(code, filename);
        let metrics = self.compute_metrics(code);
        let mut findings = self.find_defects(code, language);

        // Calculate suspiciousness using Tarantula-style scoring
        self.calculate_suspiciousness(&mut findings, &metrics);

        // Sort by suspiciousness (descending)
        findings.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limits
        if findings.len() > self.config.max_findings {
            findings.truncate(self.config.max_findings);
        }

        // Build summary
        let mut summary = HashMap::new();
        for finding in &findings {
            *summary.entry(finding.category).or_insert(0) += 1;
        }

        // Calculate health score
        let health_score = self.calculate_health_score(&findings, &metrics);

        OracleReport {
            language,
            findings,
            metrics,
            health_score,
            summary,
        }
    }

    /// Find defects using pattern matching
    fn find_defects(&self, code: &str, language: Language) -> Vec<DefectFinding> {
        let mut findings = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;

            for pattern in PATTERNS {
                // Check language applicability
                if !pattern.languages.contains(&language) && language != Language::Unknown {
                    continue;
                }

                // Check confidence threshold
                if pattern.confidence < self.config.min_confidence {
                    continue;
                }

                // Check severity filter
                if !self.config.include_info && pattern.category.severity() == 1 {
                    continue;
                }

                // Check keywords
                for keyword in pattern.keywords {
                    if let Some(col_idx) = line.find(keyword) {
                        findings.push(DefectFinding {
                            line: line_num,
                            column: col_idx + 1,
                            pattern_name: pattern.name,
                            category: pattern.category,
                            confidence: pattern.confidence,
                            suspiciousness: 0.0, // Calculated later
                            snippet: line.trim().to_string(),
                            suggestion: pattern.suggestion,
                            explanation: pattern.explanation,
                        });
                        break; // One match per pattern per line
                    }
                }
            }
        }

        // Check complexity-based patterns
        if self.config.analyze_complexity {
            self.check_complexity_patterns(code, &mut findings);
        }

        findings
    }

    /// Check for complexity-based defects
    fn check_complexity_patterns(&self, code: &str, findings: &mut Vec<DefectFinding>) {
        let lines: Vec<&str> = code.lines().collect();

        // Track nesting depth
        let mut current_depth: usize = 0;
        let mut max_depth_line = 0;
        let mut max_depth: usize = 0;

        // Track function length
        let mut in_function = false;
        let mut function_start = 0;
        let mut function_lines = 0;

        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Update nesting depth
            let open_braces = line.chars().filter(|&c| c == '{').count();
            let close_braces = line.chars().filter(|&c| c == '}').count();
            current_depth = current_depth.saturating_add(open_braces);

            if current_depth > max_depth {
                max_depth = current_depth;
                max_depth_line = idx + 1;
            }

            current_depth = current_depth.saturating_sub(close_braces);

            // Python-style nesting (indentation)
            let indent = line.len() - line.trim_start().len();
            let indent_depth = indent / 4;
            if indent_depth > max_depth {
                max_depth = indent_depth;
                max_depth_line = idx + 1;
            }

            // Track function boundaries
            if trimmed.starts_with("fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("func ")
            {
                if in_function && function_lines > 50 {
                    findings.push(DefectFinding {
                        line: function_start,
                        column: 1,
                        pattern_name: "Long Function",
                        category: DefectCategory::Complexity,
                        confidence: 0.60,
                        suspiciousness: 0.0,
                        snippet: lines
                            .get(function_start - 1)
                            .unwrap_or(&"")
                            .trim()
                            .to_string(),
                        suggestion: "Split into smaller, focused functions",
                        explanation: "Long functions are harder to understand, test, and maintain",
                    });
                }
                in_function = true;
                function_start = idx + 1;
                function_lines = 0;
            }

            if in_function {
                function_lines += 1;
            }
        }

        // Check final function
        if in_function && function_lines > 50 {
            findings.push(DefectFinding {
                line: function_start,
                column: 1,
                pattern_name: "Long Function",
                category: DefectCategory::Complexity,
                confidence: 0.60,
                suspiciousness: 0.0,
                snippet: lines
                    .get(function_start - 1)
                    .unwrap_or(&"")
                    .trim()
                    .to_string(),
                suggestion: "Split into smaller, focused functions",
                explanation: "Long functions are harder to understand, test, and maintain",
            });
        }

        // Deep nesting warning
        if max_depth > 4 {
            findings.push(DefectFinding {
                line: max_depth_line,
                column: 1,
                pattern_name: "Deep Nesting",
                category: DefectCategory::Complexity,
                confidence: 0.65,
                suspiciousness: 0.0,
                snippet: lines
                    .get(max_depth_line - 1)
                    .unwrap_or(&"")
                    .trim()
                    .to_string(),
                suggestion: "Use early returns or extract nested logic",
                explanation: "Deeply nested code is hard to follow and error-prone",
            });
        }
    }

    /// Compute code metrics
    fn compute_metrics(&self, code: &str) -> CodeMetrics {
        let lines: Vec<&str> = code.lines().collect();
        let total_lines = lines.len();

        let mut code_lines = 0;
        let mut comment_lines = 0;
        let mut max_nesting: usize = 0;
        let mut current_nesting: usize = 0;
        let mut function_count = 0;
        let mut branch_count = 0;

        for line in &lines {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            // Count comments
            if trimmed.starts_with("//")
                || trimmed.starts_with('#')
                || trimmed.starts_with("/*")
                || trimmed.starts_with('*')
            {
                comment_lines += 1;
            } else {
                code_lines += 1;
            }

            // Count functions
            if trimmed.starts_with("fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("func ")
                || trimmed.contains("=> {")
            {
                function_count += 1;
            }

            // Count branches for cyclomatic complexity
            if trimmed.starts_with("if ")
                || trimmed.starts_with("else if")
                || trimmed.starts_with("elif ")
                || trimmed.starts_with("while ")
                || trimmed.starts_with("for ")
                || trimmed.starts_with("match ")
                || trimmed.starts_with("case ")
                || trimmed.contains("&&")
                || trimmed.contains("||")
            {
                branch_count += 1;
            }

            // Track nesting
            let opens = line.chars().filter(|&c| c == '{').count();
            let closes = line.chars().filter(|&c| c == '}').count();
            current_nesting = current_nesting.saturating_add(opens);
            max_nesting = max_nesting.max(current_nesting);
            current_nesting = current_nesting.saturating_sub(closes);

            // Python indentation nesting
            let indent = line.len() - line.trim_start().len();
            max_nesting = max_nesting.max(indent / 4);
        }

        // Cyclomatic complexity = branches + 1
        let cyclomatic_complexity = branch_count + 1;

        CodeMetrics {
            total_lines,
            code_lines,
            comment_lines,
            max_nesting,
            cyclomatic_complexity,
            function_count,
        }
    }

    /// Calculate Tarantula-style suspiciousness scores
    fn calculate_suspiciousness(&self, findings: &mut [DefectFinding], metrics: &CodeMetrics) {
        // Base suspiciousness on:
        // 1. Pattern confidence
        // 2. Category severity
        // 3. Code context (nesting, complexity)

        let complexity_factor = if metrics.cyclomatic_complexity > 20 {
            1.2
        } else if metrics.cyclomatic_complexity > 10 {
            1.1
        } else {
            1.0
        };

        for finding in findings {
            let severity_weight = f32::from(finding.category.severity()) / 4.0;
            let confidence_weight = finding.confidence;

            // Tarantula formula inspired: suspiciousness = (failed/total_failed) / ((failed/total_failed) + (passed/total_passed))
            // Simplified: combine confidence and severity
            let raw_score = (confidence_weight * 0.6 + severity_weight * 0.4) * complexity_factor;

            finding.suspiciousness = raw_score.min(1.0);
        }
    }

    /// Calculate overall health score (0-100)
    fn calculate_health_score(&self, findings: &[DefectFinding], metrics: &CodeMetrics) -> u8 {
        let mut score = 100i32;

        // Deduct for findings by severity
        for finding in findings {
            match finding.category.severity() {
                4 => score -= 15, // Critical
                3 => score -= 8,  // Error
                2 => score -= 3,  // Warning
                _ => score -= 1,  // Info
            }
        }

        // Deduct for high complexity
        if metrics.cyclomatic_complexity > 30 {
            score -= 10;
        } else if metrics.cyclomatic_complexity > 20 {
            score -= 5;
        }

        // Deduct for deep nesting
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

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

fn main() {
    println!("# Demo D: Code Defect Oracle\n");
    println!("Paste code in a browser, get instant defect detection!\n");

    // Example: Analyze some problematic Rust code
    let rust_code = r#"
use std::fs::File;
use std::io::Read;

fn process_user_input(user_input: &str) -> String {
    // SQL injection risk!
    let query = format!("SELECT * FROM users WHERE name = '{}'", user_input);

    // Hardcoded credentials
    let password = "admin123";
    let api_key = "sk-1234567890";

    // Potential panic in library code
    let file = File::open("config.txt").unwrap();
    let data = some_option.unwrap();

    // Ignored error
    let _ = do_something_important();

    // Deep nesting
    if condition1 {
        if condition2 {
            if condition3 {
                if condition4 {
                    if condition5 {
                        do_deeply_nested_thing();
                    }
                }
            }
        }
    }

    query
}

fn concurrent_access() {
    static mut COUNTER: i32 = 0;

    // Data race risk
    unsafe {
        COUNTER += 1;
    }
}
"#;

    let oracle = CodeOracle::new();
    let report = oracle.analyze(rust_code, Some("example.rs"));

    println!("{}", report.format());

    println!("\n---\n");

    // Example: Analyze problematic Python code
    let python_code = r#"
import os
import subprocess

def process_data(user_input, items=[]):  # Mutable default argument!
    # Command injection risk
    os.system(f"echo {user_input}")
    subprocess.call(f"process {user_input}", shell=True)

    # SQL injection
    query = f"SELECT * FROM users WHERE id = {user_input}"

    # Bare except
    try:
        do_something()
    except:
        pass

    # Hardcoded secret
    api_key = "sk-secret-key-12345"

    # String concatenation in loop
    result = ""
    for item in items:
        result += str(item)

    return result

def get_user(user_id):
    # N+1 query pattern risk
    for user in User.objects.all():
        print(user.profile)  # Each access triggers a query
"#;

    let report = oracle.analyze(python_code, Some("example.py"));
    println!("{}", report.format());

    println!("\n---\n");

    // Example: Analyze problematic JavaScript code
    let js_code = r#"
function processUserData(userData) {
    // XSS risk!
    document.getElementById('output').innerHTML = userData;

    // eval is dangerous
    eval(userData);

    // Prototype pollution risk
    Object.assign(config, userData);

    // Loose equality
    if (userData == null) {
        return;
    }

    // SQL-like injection in query string
    const url = "api/users?filter=" + userData;

    return userData;
}

async function fetchData() {
    // Ignored promise
    fetch('/api/data');

    // No error handling
    const response = await fetch('/api/other');
}
"#;

    let report = oracle.analyze(js_code, Some("example.js"));
    println!("{}", report.format());
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Language Detection ---

    #[test]
    fn test_language_detect_rust() {
        let code = "fn main() { let x = 5; }";
        assert_eq!(Language::detect(code, Some("main.rs")), Language::Rust);
    }

    #[test]
    fn test_language_detect_python() {
        let code = "def hello(): pass";
        assert_eq!(Language::detect(code, Some("main.py")), Language::Python);
    }

    #[test]
    fn test_language_detect_javascript() {
        let code = "function hello() { return 1; }";
        assert_eq!(
            Language::detect(code, Some("main.js")),
            Language::JavaScript
        );
    }

    #[test]
    fn test_language_detect_by_content() {
        let rust_code = "fn main() -> Result<(), Error> { let x = 5; Ok(()) }";
        assert_eq!(Language::detect(rust_code, None), Language::Rust);

        let python_code = "def main():\n    x = 5\n    return x";
        assert_eq!(Language::detect(python_code, None), Language::Python);
    }

    // --- Defect Category ---

    #[test]
    fn test_category_severity() {
        assert_eq!(DefectCategory::SecurityVulnerabilities.severity(), 4);
        assert_eq!(DefectCategory::MemorySafety.severity(), 4);
        assert_eq!(DefectCategory::TypeErrors.severity(), 3);
        assert_eq!(DefectCategory::PerformanceIssues.severity(), 2);
        assert_eq!(DefectCategory::CodeSmells.severity(), 1);
    }

    #[test]
    fn test_category_name() {
        assert_eq!(DefectCategory::InjectionRisk.name(), "Injection Risk");
        assert_eq!(DefectCategory::MemorySafety.name(), "Memory Safety");
    }

    #[test]
    fn test_category_severity_label() {
        assert_eq!(
            DefectCategory::SecurityVulnerabilities.severity_label(),
            "CRITICAL"
        );
        assert_eq!(DefectCategory::TypeErrors.severity_label(), "ERROR");
        assert_eq!(
            DefectCategory::PerformanceIssues.severity_label(),
            "WARNING"
        );
        assert_eq!(DefectCategory::CodeSmells.severity_label(), "INFO");
    }

    // --- Oracle Analysis ---

    #[test]
    fn test_oracle_new() {
        let oracle = CodeOracle::new();
        assert!(oracle.config.min_confidence > 0.0);
    }

    #[test]
    fn test_oracle_detect_sql_injection() {
        let oracle = CodeOracle::new();
        let code = r#"let query = format!("SELECT * FROM users WHERE id = {}", user_id);"#;
        let report = oracle.analyze(code, Some("test.rs"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }

    #[test]
    fn test_oracle_detect_hardcoded_password() {
        let oracle = CodeOracle::new();
        let code = r#"let password = "secret123";"#;
        let report = oracle.analyze(code, Some("test.rs"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::AuthenticationIssues));
    }

    #[test]
    fn test_oracle_detect_unwrap() {
        let oracle = CodeOracle::new();
        let code = "let value = some_option.unwrap();";
        let report = oracle.analyze(code, Some("test.rs"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.pattern_name.contains("Null") || f.pattern_name.contains("Panic")));
    }

    #[test]
    fn test_oracle_detect_eval_js() {
        let oracle = CodeOracle::new();
        let code = "eval(userInput);";
        let report = oracle.analyze(code, Some("test.js"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }

    #[test]
    fn test_oracle_detect_innerhtml() {
        let oracle = CodeOracle::new();
        let code = "element.innerHTML = userData;";
        let report = oracle.analyze(code, Some("test.js"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }

    #[test]
    fn test_oracle_detect_mutable_default_python() {
        let oracle = CodeOracle::new();
        let code = "def foo(items=[]):";
        let report = oracle.analyze(code, Some("test.py"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::LogicErrors));
    }

    #[test]
    fn test_oracle_detect_shell_true() {
        let oracle = CodeOracle::new();
        let code = "subprocess.run(cmd, shell=True)";
        let report = oracle.analyze(code, Some("test.py"));

        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }

    #[test]
    fn test_oracle_clean_code() {
        let oracle = CodeOracle::new();
        let code = r#"
fn safe_function(input: &str) -> Result<String, Error> {
    let sanitized = sanitize(input);
    Ok(sanitized)
}
"#;
        let report = oracle.analyze(code, Some("test.rs"));

        // Should have high health score for clean code
        assert!(report.health_score >= 80);
    }

    #[test]
    fn test_oracle_health_score_decreases() {
        let oracle = CodeOracle::new();

        let clean_code = "fn safe() -> i32 { 42 }";
        let dirty_code = r#"
let password = "secret";
eval(user_input);
.unwrap()
"#;

        let clean_report = oracle.analyze(clean_code, Some("test.rs"));
        let dirty_report = oracle.analyze(dirty_code, Some("test.rs"));

        assert!(clean_report.health_score > dirty_report.health_score);
    }

    // --- Metrics ---

    #[test]
    fn test_metrics_line_count() {
        let oracle = CodeOracle::new();
        // Rust's .lines() doesn't count trailing newline as extra line
        let code = "line1\nline2\nline3\nline4";
        let report = oracle.analyze(code, None);

        assert_eq!(report.metrics.total_lines, 4);
    }

    #[test]
    fn test_metrics_function_count() {
        let oracle = CodeOracle::new();
        let code = "fn foo() {}\nfn bar() {}\nfn baz() {}";
        let report = oracle.analyze(code, Some("test.rs"));

        assert_eq!(report.metrics.function_count, 3);
    }

    #[test]
    fn test_metrics_comment_count() {
        let oracle = CodeOracle::new();
        let code = "// comment\ncode\n// another comment";
        let report = oracle.analyze(code, Some("test.rs"));

        assert_eq!(report.metrics.comment_lines, 2);
    }

    // --- Report ---

    #[test]
    fn test_report_format() {
        let oracle = CodeOracle::new();
        let code = r#"let password = "secret";"#;
        let report = oracle.analyze(code, Some("test.rs"));

        let formatted = report.format();
        assert!(formatted.contains("Code Defect Oracle Report"));
        assert!(formatted.contains("Health Score"));
    }

    #[test]
    fn test_finding_format() {
        let finding = DefectFinding {
            line: 10,
            column: 5,
            pattern_name: "Test Pattern",
            category: DefectCategory::SecurityVulnerabilities,
            confidence: 0.85,
            suspiciousness: 0.90,
            snippet: "dangerous code here".to_string(),
            suggestion: "Fix it",
            explanation: "Because it's bad",
        };

        let formatted = finding.format();
        assert!(formatted.contains("[CRITICAL]"));
        assert!(formatted.contains("10:5"));
        assert!(formatted.contains("Test Pattern"));
    }

    #[test]
    fn test_critical_findings() {
        let oracle = CodeOracle::new();
        let code = r#"
let password = "secret";
os.system(f"rm {user_input}");
eval(data);
"#;
        let report = oracle.analyze(code, Some("test.py"));
        let critical = report.critical_findings();

        // Should have some critical findings
        assert!(!critical.is_empty() || report.findings.is_empty());
    }

    // --- Config ---

    #[test]
    fn test_config_default() {
        let config = OracleConfig::default();
        assert!(config.min_confidence > 0.0);
        assert!(config.max_findings > 0);
    }

    #[test]
    fn test_config_custom() {
        let config = OracleConfig {
            min_confidence: 0.8,
            max_findings: 10,
            include_info: true,
            analyze_complexity: false,
        };
        let oracle = CodeOracle::with_config(config);

        let code = "let x = 1 == 2;"; // Magic number (low confidence)
        let report = oracle.analyze(code, Some("test.rs"));

        // With high min_confidence, low-confidence patterns shouldn't match
        assert!(report.findings.is_empty() || report.findings[0].confidence >= 0.8);
    }

    // --- Suspiciousness ---

    #[test]
    fn test_suspiciousness_bounded() {
        let oracle = CodeOracle::new();
        let code = r#"
let password = "secret";
eval(user);
.unwrap()
"#;
        let report = oracle.analyze(code, Some("test.rs"));

        for finding in &report.findings {
            assert!(finding.suspiciousness >= 0.0);
            assert!(finding.suspiciousness <= 1.0);
        }
    }

    #[test]
    fn test_findings_sorted_by_suspiciousness() {
        let oracle = CodeOracle::new();
        let code = r#"
let password = "secret";
let x = 1;
eval(user);
"#;
        let report = oracle.analyze(code, Some("test.js"));

        if report.findings.len() > 1 {
            for i in 1..report.findings.len() {
                assert!(report.findings[i - 1].suspiciousness >= report.findings[i].suspiciousness);
            }
        }
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_language_detect_never_crashes(code in ".*") {
            let _ = Language::detect(&code, None);
        }

        #[test]
        fn prop_oracle_never_crashes(code in ".*") {
            let oracle = CodeOracle::new();
            let _ = oracle.analyze(&code, None);
        }

        #[test]
        fn prop_health_score_bounded(code in ".*") {
            let oracle = CodeOracle::new();
            let report = oracle.analyze(&code, None);
            prop_assert!(report.health_score <= 100);
        }

        #[test]
        fn prop_suspiciousness_bounded(code in "[a-zA-Z0-9 .unwrap()eval()password=\"test\"]{0,500}") {
            let oracle = CodeOracle::new();
            let report = oracle.analyze(&code, None);

            for finding in &report.findings {
                prop_assert!(finding.suspiciousness >= 0.0);
                prop_assert!(finding.suspiciousness <= 1.0);
                prop_assert!(finding.confidence >= 0.0);
                prop_assert!(finding.confidence <= 1.0);
            }
        }

        #[test]
        fn prop_metrics_non_negative(code in ".*") {
            let oracle = CodeOracle::new();
            let report = oracle.analyze(&code, None);

            // total_lines is usize, always >= 0, so just check relational invariant
            prop_assert!(report.metrics.code_lines <= report.metrics.total_lines);
        }

        #[test]
        fn prop_findings_count_respects_limit(
            code in "[a-zA-Z0-9 .unwrap()eval()password=\"]{0,200}",
            max_findings in 1usize..20
        ) {
            let config = OracleConfig {
                max_findings,
                ..Default::default()
            };
            let oracle = CodeOracle::with_config(config);
            let report = oracle.analyze(&code, None);

            prop_assert!(report.findings.len() <= max_findings);
        }

        #[test]
        fn prop_severity_valid(severity in 1u8..=4) {
            // All categories should have severity 1-4
            let categories = [
                DefectCategory::MemorySafety,
                DefectCategory::SecurityVulnerabilities,
                DefectCategory::TypeErrors,
                DefectCategory::CodeSmells,
            ];
            for cat in categories {
                prop_assert!(cat.severity() >= 1);
                prop_assert!(cat.severity() <= 4);
            }
            let _ = severity; // Use the generated value
        }

        #[test]
        fn prop_report_format_not_empty(code in ".{0,100}") {
            let oracle = CodeOracle::new();
            let report = oracle.analyze(&code, None);
            let formatted = report.format();

            prop_assert!(!formatted.is_empty());
            prop_assert!(formatted.contains("Code Defect Oracle"));
        }

        #[test]
        fn prop_pattern_confidence_valid(_idx in 0usize..PATTERNS.len()) {
            for pattern in PATTERNS {
                prop_assert!(pattern.confidence >= 0.0);
                prop_assert!(pattern.confidence <= 1.0);
            }
        }
    }
}
