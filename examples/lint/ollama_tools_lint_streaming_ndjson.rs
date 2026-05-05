//! # Recipe: Ollama Tools Lint — Streaming NDJSON
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-tools-lint --response-file resp.ndjson --stream`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the NDJSON-stream parser inside `apr ollama-tools-lint`.
//! When `--stream` is set the file is parsed line-by-line; each line must be
//! a complete JSON object (newline-delimited JSON) and only the **last**
//! frame may have `done == true`. The lint flags four classes of stream
//! defect: empty stream, mid-stream done=true, malformed JSON line, and
//! conflicting model field across frames (which would mean two streams got
//! interleaved into one file).
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_tools_lint_streaming_ndjson
//! ```
//!
//! ## References
//! - aprender CRUX-I-04 (streaming invariant).
//! - NDJSON 1.0 (ndjson.org).
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamFinding {
    EmptyStream,
    MidStreamDone { line: usize },
    MalformedJson { line: usize, snippet: String },
    ModelMismatch { models: Vec<String> },
}

pub fn lint_stream(ndjson: &str) -> Vec<StreamFinding> {
    let lines: Vec<&str> = ndjson.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return vec![StreamFinding::EmptyStream];
    }

    let mut out = Vec::new();
    let mut models: HashSet<String> = HashSet::new();
    let mut frames: Vec<Value> = Vec::with_capacity(lines.len());

    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str::<Value>(line) {
            Ok(v) => {
                if let Some(m) = v.get("model").and_then(Value::as_str) {
                    models.insert(m.into());
                }
                frames.push(v);
            }
            Err(_) => out.push(StreamFinding::MalformedJson {
                line: i,
                snippet: line.chars().take(40).collect(),
            }),
        }
    }

    if models.len() > 1 {
        let mut sorted: Vec<String> = models.into_iter().collect();
        sorted.sort();
        out.push(StreamFinding::ModelMismatch { models: sorted });
    }

    let last_idx = frames.len().saturating_sub(1);
    for (i, f) in frames.iter().enumerate() {
        let done = f.get("done").and_then(Value::as_bool) == Some(true);
        if done && i != last_idx {
            out.push(StreamFinding::MidStreamDone { line: i });
        }
    }

    out
}

fn build_clean_stream() -> String {
    [
        r#"{"model":"llama3.1:8b","done":false,"message":{"role":"assistant","content":"Hello"}}"#,
        r#"{"model":"llama3.1:8b","done":false,"message":{"role":"assistant","content":" world"}}"#,
        r#"{"model":"llama3.1:8b","done":true, "message":{"role":"assistant","content":""}}"#,
    ]
    .join("\n")
}

fn build_corrupt_stream() -> String {
    [
        r#"{"model":"llama3.1:8b","done":true, "message":{"role":"assistant"}}"#, // mid-stream done
        r#"{"model":"qwen2.5:7b", "done":false,"message":{"role":"assistant"}}"#, // model mismatch
        "{not json}",                                                             // malformed
        r#"{"model":"llama3.1:8b","done":true, "message":{"role":"assistant"}}"#,
    ]
    .join("\n")
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_tools_lint_streaming_ndjson")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("clean stream:   {:?}", lint_stream(&build_clean_stream()));
    println!("corrupt stream: {:?}", lint_stream(&build_corrupt_stream()));

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn clean_stream_has_no_findings() {
        let f = lint_stream(&build_clean_stream());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn empty_stream_yields_empty_finding() {
        let f = lint_stream("");
        assert_eq!(f, vec![StreamFinding::EmptyStream]);
    }

    #[test]
    fn whitespace_only_stream_is_empty() {
        let f = lint_stream("\n\n   \n\n");
        assert_eq!(f, vec![StreamFinding::EmptyStream]);
    }

    #[test]
    fn mid_stream_done_flagged() {
        let f = lint_stream(&build_corrupt_stream());
        assert!(f
            .iter()
            .any(|x| matches!(x, StreamFinding::MidStreamDone { .. })));
    }

    #[test]
    fn model_mismatch_flagged() {
        let f = lint_stream(&build_corrupt_stream());
        assert!(f
            .iter()
            .any(|x| matches!(x, StreamFinding::ModelMismatch { .. })));
    }

    #[test]
    fn malformed_json_flagged() {
        let f = lint_stream(&build_corrupt_stream());
        assert!(f
            .iter()
            .any(|x| matches!(x, StreamFinding::MalformedJson { .. })));
    }
}
