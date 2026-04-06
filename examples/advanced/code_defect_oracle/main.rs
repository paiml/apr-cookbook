#![allow(unused_imports)]
//! # Demo D: Code Defect Oracle
//! **CLI Equivalent**: `apr code`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Browser-ready code analysis tool detecting defects, security issues, and anti-patterns.
//! Multi-language support (Rust, Python, JavaScript, Go) with 18 defect categories,
//! Tarantula-style fault localization, and actionable fix suggestions.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::collections::HashMap;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("# Demo D: Code Defect Oracle\n");
    let oracle = CodeOracle::new();
    let rust_code = r#"
fn process_user_input(user_input: &str) -> String {
    let query = format!("SELECT * FROM users WHERE name = '{}'", user_input);
    let password = "admin123";
    let file = File::open("config.txt").unwrap();
    let _ = do_something_important();
    if c1 { if c2 { if c3 { if c4 { if c5 { do_thing(); } } } } }
    query
}
"#;
    println!("{}", oracle.analyze(rust_code, Some("example.rs")).format());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detect_by_extension() {
        assert_eq!(Language::detect("", Some("main.rs")), Language::Rust);
        assert_eq!(Language::detect("", Some("main.py")), Language::Python);
        assert_eq!(Language::detect("", Some("main.js")), Language::JavaScript);
    }
    #[test]
    fn test_language_detect_by_content() {
        assert_eq!(
            Language::detect("fn main() -> Result<(), Error> { let x = 5; Ok(()) }", None),
            Language::Rust
        );
        assert_eq!(
            Language::detect("def main():\n    x = 5\n    return x", None),
            Language::Python
        );
    }
    #[test]
    fn test_category_severity() {
        assert_eq!(DefectCategory::SecurityVulnerabilities.severity(), 4);
        assert_eq!(DefectCategory::TypeErrors.severity(), 3);
        assert_eq!(DefectCategory::PerformanceIssues.severity(), 2);
        assert_eq!(DefectCategory::CodeSmells.severity(), 1);
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
    #[test]
    fn test_detect_sql_injection() {
        let report = CodeOracle::new().analyze(
            r#"let q = format!("SELECT * FROM users WHERE id = {}", id);"#,
            Some("t.rs"),
        );
        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }
    #[test]
    fn test_detect_hardcoded_password() {
        let report = CodeOracle::new().analyze(r#"let password = "secret123";"#, Some("t.rs"));
        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::AuthenticationIssues));
    }
    #[test]
    fn test_detect_unwrap() {
        let report = CodeOracle::new().analyze("let v = x.unwrap();", Some("t.rs"));
        assert!(report
            .findings
            .iter()
            .any(|f| f.pattern_name.contains("Null") || f.pattern_name.contains("Panic")));
    }
    #[test]
    fn test_detect_eval_js() {
        let report = CodeOracle::new().analyze("eval(userInput);", Some("t.js"));
        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }
    #[test]
    fn test_detect_shell_true() {
        let report = CodeOracle::new().analyze("subprocess.run(cmd, shell=True)", Some("t.py"));
        assert!(report
            .findings
            .iter()
            .any(|f| f.category == DefectCategory::InjectionRisk));
    }
    #[test]
    fn test_health_score_clean_vs_dirty() {
        let oracle = CodeOracle::new();
        let clean = oracle.analyze("fn safe() -> i32 { 42 }", Some("t.rs"));
        let dirty = oracle.analyze("let password = \"s\";\neval(u);\n.unwrap()", Some("t.rs"));
        assert!(clean.health_score > dirty.health_score);
    }
    #[test]
    fn test_metrics() {
        let oracle = CodeOracle::new();
        let r = oracle.analyze("fn foo() {}\nfn bar() {}\nfn baz() {}", Some("t.rs"));
        assert_eq!(r.metrics.function_count, 3);
        assert_eq!(r.metrics.total_lines, 3);
    }
    #[test]
    fn test_report_format() {
        let report = CodeOracle::new().analyze(r#"let password = "secret";"#, Some("t.rs"));
        let fmt = report.format();
        assert!(fmt.contains("Code Defect Oracle Report"));
        assert!(fmt.contains("Health Score"));
    }
    #[test]
    fn test_config_custom_high_confidence() {
        let oracle = CodeOracle::with_config(OracleConfig {
            min_confidence: 0.8,
            max_findings: 10,
            include_info: true,
            analyze_complexity: false,
        });
        let report = oracle.analyze("let x = 1 == 2;", Some("t.rs"));
        assert!(report.findings.is_empty() || report.findings[0].confidence >= 0.8);
    }
    #[test]
    fn test_suspiciousness_bounded() {
        let report =
            CodeOracle::new().analyze("let password = \"s\";\neval(u);\n.unwrap()", Some("t.rs"));
        for f in &report.findings {
            assert!((0.0..=1.0).contains(&f.suspiciousness));
        }
    }
    #[test]
    fn test_findings_sorted() {
        let report = CodeOracle::new().analyze("let password = \"s\";\neval(u);", Some("t.js"));
        if report.findings.len() > 1 {
            for i in 1..report.findings.len() {
                assert!(report.findings[i - 1].suspiciousness >= report.findings[i].suspiciousness);
            }
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test] fn prop_oracle_never_crashes(code in ".*") { let _ = CodeOracle::new().analyze(&code, None); }
        #[test] fn prop_health_score_bounded(code in ".*") { prop_assert!(CodeOracle::new().analyze(&code, None).health_score <= 100); }
        #[test] fn prop_suspiciousness_bounded(code in "[a-zA-Z0-9 .unwrap()eval()password=\"test\"]{0,500}") {
            for f in &CodeOracle::new().analyze(&code, None).findings { prop_assert!((0.0..=1.0).contains(&f.suspiciousness)); prop_assert!((0.0..=1.0).contains(&f.confidence)); }
        }
        #[test] fn prop_findings_respect_limit(code in "[a-zA-Z0-9 .unwrap()eval()password=\"]{0,200}", max in 1usize..20) {
            let oracle = CodeOracle::with_config(OracleConfig { max_findings: max, ..Default::default() });
            prop_assert!(oracle.analyze(&code, None).findings.len() <= max);
        }
        #[test] fn prop_pattern_confidence_valid(_idx in 0usize..PATTERNS.len()) {
            for p in PATTERNS { prop_assert!(p.confidence >= 0.0 && p.confidence <= 1.0); }
        }
    }
}
