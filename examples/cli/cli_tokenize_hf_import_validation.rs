//! # apr tokenize import-hf — Two-File Layout Validation
//!
//! `apr tokenize import-hf <FILE>` ingests a Hugging Face `tokenizer.json`
//! and emits aprender's two-file layout: `vocab.json` + `merges.txt`. This
//! recipe validates the canonicalisation: the input must declare a BPE
//! model, the vocab map must be a 1:1 mapping (no duplicates), and the
//! merges array must be flat-key with no duplicate pairs.
//!
//! Demonstrates the **TOKENIZE.4** recipe for PMAT-095 (apr tokenize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts/apr-cli-tokenize-import-hf-v1.yaml
//!
//! Run with: cargo run --example cli_tokenize_hf_import_validation
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq)]
pub enum ImportFinding {
    NotBpe { observed: String },
    DuplicateVocabId { id: u64 },
    DuplicateMerge { merge: String },
    MalformedMergeLine { line: usize, content: String },
    EmptyVocab,
    EmptyMerges,
}

pub fn validate_hf_tokenizer(t: &Value) -> Vec<ImportFinding> {
    let mut out = Vec::new();

    let model_type = t
        .pointer("/model/type")
        .and_then(Value::as_str)
        .unwrap_or("");
    if model_type != "BPE" {
        out.push(ImportFinding::NotBpe {
            observed: model_type.to_string(),
        });
    }

    let vocab = t.pointer("/model/vocab").and_then(Value::as_object);
    let merges = t.pointer("/model/merges").and_then(Value::as_array);

    match vocab {
        Some(v) if !v.is_empty() => {
            let mut seen: HashSet<u64> = HashSet::new();
            for (_, val) in v {
                let id = val.as_u64().unwrap_or(0);
                if !seen.insert(id) {
                    out.push(ImportFinding::DuplicateVocabId { id });
                }
            }
        }
        Some(_) => out.push(ImportFinding::EmptyVocab),
        None => out.push(ImportFinding::EmptyVocab),
    }

    match merges {
        Some(m) if !m.is_empty() => {
            let mut seen: HashSet<String> = HashSet::new();
            for (i, line) in m.iter().enumerate() {
                let s = line.as_str().unwrap_or("");
                let parts: Vec<&str> = s.split_whitespace().collect();
                if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
                    out.push(ImportFinding::MalformedMergeLine {
                        line: i,
                        content: s.into(),
                    });
                    continue;
                }
                if !seen.insert(s.into()) {
                    out.push(ImportFinding::DuplicateMerge { merge: s.into() });
                }
            }
        }
        Some(_) => out.push(ImportFinding::EmptyMerges),
        None => out.push(ImportFinding::EmptyMerges),
    }

    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tokenize_hf_import_validation")?;

    let happy = json!({
        "model": {
            "type": "BPE",
            "vocab": { "<pad>": 0, "<eos>": 1, "hello": 2, "world": 3 },
            "merges": ["h e", "l l", "he ll", "hell o", "wor ld"]
        }
    });
    let bad = json!({
        "model": {
            "type": "WordLevel",
            "vocab": {},
            "merges": ["bogus", "h e", "h e"]
        }
    });

    println!("happy:  {:?}", validate_hf_tokenizer(&happy));
    println!("bad:    {:?}", validate_hf_tokenizer(&bad));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_tokenizer_passes() {
        let t = json!({
            "model": {
                "type": "BPE",
                "vocab": { "a": 0, "b": 1 },
                "merges": ["a b"]
            }
        });
        assert!(validate_hf_tokenizer(&t).is_empty());
    }

    #[test]
    fn non_bpe_model_rejected() {
        // SentencePiece, Unigram, WordLevel etc. need different importers.
        let t = json!({
            "model": { "type": "Unigram", "vocab": { "a": 0 }, "merges": ["a b"] }
        });
        let f = validate_hf_tokenizer(&t);
        assert!(f.iter().any(|x| matches!(x, ImportFinding::NotBpe { .. })));
    }

    #[test]
    fn duplicate_vocab_id_flagged() {
        // Two tokens mapped to same id breaks the inverse-vocab lookup.
        let t = json!({
            "model": { "type": "BPE", "vocab": { "a": 0, "b": 0 }, "merges": ["a b"] }
        });
        let f = validate_hf_tokenizer(&t);
        assert!(f
            .iter()
            .any(|x| matches!(x, ImportFinding::DuplicateVocabId { id: 0 })));
    }

    #[test]
    fn malformed_merge_line_flagged() {
        // BPE merges are space-separated pairs; "bogus" has no pair.
        let t = json!({
            "model": { "type": "BPE", "vocab": { "a": 0 }, "merges": ["bogus"] }
        });
        let f = validate_hf_tokenizer(&t);
        assert!(f
            .iter()
            .any(|x| matches!(x, ImportFinding::MalformedMergeLine { line: 0, .. })));
    }

    #[test]
    fn duplicate_merge_flagged() {
        let t = json!({
            "model": { "type": "BPE", "vocab": { "a": 0 }, "merges": ["a b", "a b"] }
        });
        let f = validate_hf_tokenizer(&t);
        assert!(f
            .iter()
            .any(|x| matches!(x, ImportFinding::DuplicateMerge { .. })));
    }

    #[test]
    fn empty_vocab_flagged() {
        let t = json!({
            "model": { "type": "BPE", "vocab": {}, "merges": ["a b"] }
        });
        let f = validate_hf_tokenizer(&t);
        assert!(f.iter().any(|x| x == &ImportFinding::EmptyVocab));
    }
}
