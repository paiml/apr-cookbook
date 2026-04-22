//! # Recipe: Tool-Use Lint — Streaming Assembly Pipeline
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr tool-use-lint stream.ndjson --stream`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example tool_use_lint_streaming` exits 0
//! 2. [x] `cargo test --example tool_use_lint_streaming` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Tool-use responses over the streaming API split the JSON argument object
//! across multiple SSE-style fragments. This recipe assembles the fragments
//! back into one string, validates the reassembled JSON, and confirms that
//! the streaming sequence terminates with `finish_reason == "tool_calls"`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tool_use_lint_streaming
//! ```
//!
//! ## References
//! - Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct StreamReport {
    pub fragments: usize,
    pub assembled_args: String,
    pub finish_reason: Option<String>,
    pub findings: Vec<String>,
}

pub fn assemble(stream: &str) -> StreamReport {
    let mut fragments = 0usize;
    let mut assembled = String::new();
    let mut finish_reason: Option<String> = None;
    let mut findings = Vec::new();
    for (i, line) in stream.lines().filter(|l| !l.is_empty()).enumerate() {
        fragments += 1;
        let Ok(v): std::result::Result<Value, _> = serde_json::from_str(line) else {
            findings.push(format!("non-JSON fragment at index {i}"));
            continue;
        };
        if let Some(arg_frag) = v
            .pointer("/choices/0/delta/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
        {
            assembled.push_str(arg_frag);
        }
        if let Some(fr) = v
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            finish_reason = Some(fr.to_string());
        }
    }
    match serde_json::from_str::<Value>(&assembled) {
        Ok(v) if v.is_object() => {}
        _ => findings.push(format!(
            "assembled arguments not a valid JSON object: {assembled:?}"
        )),
    }
    if finish_reason.as_deref() != Some("tool_calls") {
        findings.push(format!(
            "stream did not terminate with finish_reason=tool_calls (got {:?})",
            finish_reason
        ));
    }
    StreamReport {
        fragments,
        assembled_args: assembled,
        finish_reason,
        findings,
    }
}

fn build_stream() -> String {
    let frags = [
        (r#"{"city":"#, None),
        (r#""Tokyo","#, None),
        (r#""units":"metric"}"#, None),
        ("", Some("tool_calls")),
    ];
    let mut s = String::new();
    for (arg_frag, fr) in frags {
        let mut v = json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": arg_frag}
                    }]
                }
            }]
        });
        if let Some(fr) = fr {
            v["choices"][0]["finish_reason"] = json!(fr);
        }
        s.push_str(&serde_json::to_string(&v).unwrap_or_default());
        s.push('\n');
    }
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("tool_use_lint_streaming")?;
    let stream = build_stream();
    let p = ctx.path("stream.ndjson");
    std::fs::write(&p, &stream)?;

    let rep = assemble(&stream);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Fragments: {}", rep.fragments);
    println!("Assembled: {:?}", rep.assembled_args);
    println!("finish_reason: {:?}", rep.finish_reason);
    if rep.findings.is_empty() {
        println!("Verdict: PASS");
    } else {
        for f in &rep.findings {
            println!("  {f}");
        }
    }

    ctx.record_metric("fragments", rep.fragments as i64);
    ctx.record_metric("findings", rep.findings.len() as i64);
    ctx.record_string_metric(
        "verdict",
        if rep.findings.is_empty() {
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
    fn stream_assembles_into_valid_json() {
        let r = assemble(&build_stream());
        assert!(r.findings.is_empty(), "{:?}", r.findings);
        let v: Value = serde_json::from_str(&r.assembled_args).expect("parse");
        assert_eq!(v["city"], "Tokyo");
        assert_eq!(v["units"], "metric");
    }

    #[test]
    fn missing_finish_reason_flags_finding() {
        let mut s = build_stream();
        s = s
            .lines()
            .filter(|l| !l.contains("finish_reason"))
            .collect::<Vec<_>>()
            .join("\n");
        s.push('\n');
        let r = assemble(&s);
        assert!(r.findings.iter().any(|x| x.contains("finish_reason")));
    }

    #[test]
    fn fragment_count_is_four() {
        let r = assemble(&build_stream());
        assert_eq!(r.fragments, 4);
    }
}
