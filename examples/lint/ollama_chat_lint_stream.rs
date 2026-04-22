//! # Recipe: Ollama-Chat Lint — Streaming NDJSON
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-chat-lint stream.ndjson --stream`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example ollama_chat_lint_stream` exits 0
//! 2. [x] `cargo test --example ollama_chat_lint_stream` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the NDJSON streaming variant of `/api/chat`: one JSON object
//! per line, with `done: false` on every intermediate chunk and `done: true`
//! on the terminator. Enforces: exactly one terminator, terminator is last,
//! token fragments concatenate into a non-empty transcript.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_chat_lint_stream
//! ```
//!
//! ## References
//! - Ollama. *API Reference: /api/chat (streaming)*. <https://github.com/ollama/ollama/blob/main/docs/api.md>
//! - Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct StreamReport {
    pub chunks: usize,
    pub terminator_index: Option<usize>,
    pub transcript: String,
    pub findings: Vec<String>,
}

pub fn lint_stream(ndjson: &str) -> StreamReport {
    let mut chunks = 0usize;
    let mut terminator_index = None;
    let mut transcript = String::new();
    let mut findings = Vec::new();
    for (i, line) in ndjson.lines().filter(|l| !l.is_empty()).enumerate() {
        chunks = i + 1;
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                findings.push(format!("OLL-100: non-JSON line {i}: {e}"));
                continue;
            }
        };
        if let Some(frag) = v.pointer("/message/content").and_then(Value::as_str) {
            transcript.push_str(frag);
        }
        if v.get("done").and_then(Value::as_bool) == Some(true) {
            if terminator_index.is_some() {
                findings.push(format!("OLL-101: multiple terminators (extra at {i})"));
            }
            terminator_index = Some(i);
        }
    }
    match terminator_index {
        None => findings.push("OLL-102: stream lacks `done: true` terminator".into()),
        Some(idx) if idx + 1 != chunks => {
            findings.push(format!(
                "OLL-103: terminator at index {idx} is not last (stream len {chunks})"
            ));
        }
        _ => {}
    }
    if transcript.is_empty() {
        findings.push("OLL-104: transcript empty after concatenation".into());
    }
    StreamReport {
        chunks,
        terminator_index,
        transcript,
        findings,
    }
}

fn build_stream() -> String {
    let mut s = String::new();
    for chunk in ["Hello", ", ", "world", "!"] {
        let v = json!({
            "model": "llama3.1:8b",
            "created_at": "2026-04-22T12:00:00Z",
            "message": {"role": "assistant", "content": chunk},
            "done": false
        });
        s.push_str(&serde_json::to_string(&v).unwrap_or_default());
        s.push('\n');
    }
    let tail = json!({
        "model": "llama3.1:8b",
        "created_at": "2026-04-22T12:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": true,
        "total_duration": 42_000_000u64
    });
    s.push_str(&serde_json::to_string(&tail).unwrap_or_default());
    s.push('\n');
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_chat_lint_stream")?;
    let ndjson = build_stream();
    let p = ctx.path("stream.ndjson");
    std::fs::write(&p, &ndjson)?;

    let report = lint_stream(&ndjson);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Stream: {}", p.display());
    println!("Chunks: {}", report.chunks);
    println!(
        "Terminator at: {:?} / {}",
        report.terminator_index, report.chunks
    );
    println!("Transcript: {:?}", report.transcript);
    if report.findings.is_empty() {
        println!("Verdict: PASS");
    } else {
        println!("Findings:");
        for f in &report.findings {
            println!("  {f}");
        }
    }

    ctx.record_metric("chunks", report.chunks as i64);
    ctx.record_metric("findings", report.findings.len() as i64);
    ctx.record_string_metric("transcript", &report.transcript);
    ctx.record_string_metric(
        "verdict",
        if report.findings.is_empty() {
            "PASS"
        } else {
            "FAIL"
        },
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_stream_is_clean() {
        let r = lint_stream(&build_stream());
        assert!(r.findings.is_empty(), "{:?}", r.findings);
        assert_eq!(r.transcript, "Hello, world!");
    }

    #[test]
    fn missing_terminator_flags_oll_102() {
        let mut s = build_stream();
        // drop the last line (the terminator)
        s = s
            .lines()
            .take_while(|l| !l.contains("\"done\":true"))
            .collect::<Vec<_>>()
            .join("\n");
        s.push('\n');
        let r = lint_stream(&s);
        assert!(r.findings.iter().any(|f| f.starts_with("OLL-102")));
    }

    #[test]
    fn terminator_not_last_flags_oll_103() {
        let mut s = build_stream();
        s.push_str(r#"{"model":"x","message":{"content":"trailer"},"done":false}"#);
        s.push('\n');
        let r = lint_stream(&s);
        assert!(r.findings.iter().any(|f| f.starts_with("OLL-103")));
    }
}
