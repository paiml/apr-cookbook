//! # apr diagnose — JSONL Corpus Validator
//!
//! `apr diagnose <DIR> --data <FILE.jsonl>` requires a JSONL test corpus.
//! This recipe validates the corpus shape before invocation: every line
//! must parse as JSON, contain the expected keys (`text`, `label`),
//! `label` must be in `[0, num_classes)`, and the corpus must contain at
//! least one example of each class (otherwise per-class metrics are
//! undefined).
//!
//! Demonstrates the **DIAGNOSE.4** recipe for PMAT-095 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAGNOSE-002 + JSONL spec (jsonlines.org)
//!
//! Run with: cargo run --example cli_diagnose_jsonl_corpus_validator
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Eq)]
pub enum CorpusFinding {
    MalformedLine { line: usize },
    MissingKey { line: usize, key: &'static str },
    LabelOutOfRange { line: usize, label: i64 },
    EmptyText { line: usize },
    UnusedClass { class: u32 },
    EmptyCorpus,
}

pub fn validate_corpus(jsonl: &str, num_classes: u32) -> Vec<CorpusFinding> {
    let mut out = Vec::new();
    let lines: Vec<&str> = jsonl.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return vec![CorpusFinding::EmptyCorpus];
    }
    let mut seen_classes: BTreeSet<u32> = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            out.push(CorpusFinding::MalformedLine { line: i });
            continue;
        };
        let text = v.get("text").and_then(Value::as_str);
        let label = v.get("label").and_then(Value::as_i64);
        if text.is_none() {
            out.push(CorpusFinding::MissingKey {
                line: i,
                key: "text",
            });
        }
        if label.is_none() {
            out.push(CorpusFinding::MissingKey {
                line: i,
                key: "label",
            });
        }
        if let Some(t) = text {
            if t.is_empty() {
                out.push(CorpusFinding::EmptyText { line: i });
            }
        }
        if let Some(l) = label {
            if l < 0 || l >= i64::from(num_classes) {
                out.push(CorpusFinding::LabelOutOfRange { line: i, label: l });
            } else {
                seen_classes.insert(l as u32);
            }
        }
    }
    for c in 0..num_classes {
        if !seen_classes.contains(&c) {
            out.push(CorpusFinding::UnusedClass { class: c });
        }
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_jsonl_corpus_validator")?;

    let happy = "\
{\"text\":\"hello world\",\"label\":0}
{\"text\":\"good morning\",\"label\":1}
{\"text\":\"farewell\",\"label\":2}";

    let dirty = "\
{\"text\":\"good\",\"label\":0}
not json
{\"text\":\"\",\"label\":1}
{\"text\":\"bad\",\"label\":99}";

    println!("happy:  {:?}", validate_corpus(happy, 3));
    println!("dirty:  {:?}", validate_corpus(dirty, 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_corpus_is_clean() {
        let c = "\
{\"text\":\"a\",\"label\":0}
{\"text\":\"b\",\"label\":1}
{\"text\":\"c\",\"label\":2}";
        assert!(validate_corpus(c, 3).is_empty());
    }

    #[test]
    fn empty_corpus_yields_single_finding() {
        let f = validate_corpus("", 3);
        assert_eq!(f, vec![CorpusFinding::EmptyCorpus]);
    }

    #[test]
    fn malformed_line_flagged() {
        let c = "{\"text\":\"a\",\"label\":0}\nnot json\n{\"text\":\"b\",\"label\":1}\n{\"text\":\"c\",\"label\":2}";
        let f = validate_corpus(c, 3);
        assert!(f
            .iter()
            .any(|x| matches!(x, CorpusFinding::MalformedLine { line: 1 })));
    }

    #[test]
    fn label_out_of_range_flagged() {
        let c = "{\"text\":\"a\",\"label\":99}\n{\"text\":\"b\",\"label\":1}\n{\"text\":\"c\",\"label\":2}";
        let f = validate_corpus(c, 3);
        assert!(f
            .iter()
            .any(|x| matches!(x, CorpusFinding::LabelOutOfRange { label: 99, .. })));
    }

    #[test]
    fn unused_class_flagged() {
        // 3 classes declared but corpus only covers 0 and 1 — class 2 unused.
        let c = "{\"text\":\"a\",\"label\":0}\n{\"text\":\"b\",\"label\":1}";
        let f = validate_corpus(c, 3);
        assert!(f
            .iter()
            .any(|x| x == &CorpusFinding::UnusedClass { class: 2 }));
    }

    #[test]
    fn empty_text_flagged_separately_from_missing() {
        let c = "{\"text\":\"\",\"label\":0}\n{\"text\":\"b\",\"label\":1}\n{\"text\":\"c\",\"label\":2}";
        let f = validate_corpus(c, 3);
        assert!(f
            .iter()
            .any(|x| matches!(x, CorpusFinding::EmptyText { line: 0 })));
        // No MissingKey because the field is present, just empty.
        assert!(!f
            .iter()
            .any(|x| matches!(x, CorpusFinding::MissingKey { line: 0, .. })));
    }
}
