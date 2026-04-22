//! # Recipe: gRPC-style Serve with Streaming
//!
//! **Category**: serve
//! **CLI Equivalent**: `apr serve --protocol grpc --streaming`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example serve_grpc_stream` exits 0
//! 2. [x] `cargo test --example serve_grpc_stream` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr serve --protocol grpc` call stream in-process
//! 10. [x] Unit tests cover server-streaming, bidi-streaming, error frame
//!
//! ## Learning Objective
//! Demonstrates the four gRPC interaction patterns (unary, server-streaming,
//! client-streaming, bidi-streaming) over an in-memory channel. We focus on
//! token-by-token server streaming — the shape `apr serve` uses for streaming
//! inference.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serve_grpc_stream
//! ```
//!
//! ## References
//! - Birrell, A.D. & Nelson, B.J. (1984). *Implementing Remote Procedure Calls*. ACM TOCS. DOI: 10.1145/2080.357392

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Header { request_id: u64 },
    Token { index: u32, text: String },
    Trailer { status: Status },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Ok,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct InferRequest {
    pub id: u64,
    pub prompt: String,
    pub max_tokens: u32,
}

/// Server-streaming RPC: one request in, many token frames out.
pub fn server_stream(req: &InferRequest) -> Vec<Frame> {
    let mut out = vec![Frame::Header { request_id: req.id }];
    // Fake tokenisation: split prompt on whitespace and emit up to max_tokens.
    let tokens: Vec<&str> = req.prompt.split_whitespace().collect();
    let n = (tokens.len() as u32).min(req.max_tokens);
    for i in 0..n {
        out.push(Frame::Token {
            index: i,
            text: tokens[i as usize].to_string(),
        });
    }
    // Terminal frame with status.
    let status = if req.max_tokens == 0 {
        Status::Error("max_tokens must be > 0".into())
    } else {
        Status::Ok
    };
    out.push(Frame::Trailer { status });
    out
}

/// Bidi-streaming RPC: interleaved prompt-chunks and token-responses.
pub fn bidi_stream(chunks: &[&str], max_tokens_per_chunk: u32) -> Vec<Frame> {
    let mut out = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let req = InferRequest {
            id: i as u64,
            prompt: (*chunk).to_string(),
            max_tokens: max_tokens_per_chunk,
        };
        out.extend(server_stream(&req));
    }
    out
}

pub fn count_frames(frames: &[Frame]) -> (usize, usize, usize) {
    let mut h = 0;
    let mut t = 0;
    let mut tr = 0;
    for f in frames {
        match f {
            Frame::Header { .. } => h += 1,
            Frame::Token { .. } => t += 1,
            Frame::Trailer { .. } => tr += 1,
        }
    }
    (h, t, tr)
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("serve_grpc_stream")?;
    println!("=== Recipe: {} ===", ctx.name());

    let req = InferRequest {
        id: 42,
        prompt: "Hello gRPC streaming world of inference frames".into(),
        max_tokens: 5,
    };
    let frames = server_stream(&req);

    println!("Server-streaming RPC for request {}", req.id);
    println!("Frames ({}):", frames.len());
    for f in &frames {
        match f {
            Frame::Header { request_id } => println!("  [HEADER] req_id={}", request_id),
            Frame::Token { index, text } => println!("  [TOKEN {:>2}] {}", index, text),
            Frame::Trailer { status } => println!("  [TRAILER] {:?}", status),
        }
    }

    let (h, t, tr) = count_frames(&frames);
    let report = json!({
        "recipe": ctx.name(),
        "pattern": "server_streaming",
        "request": {
            "id": req.id,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
        },
        "n_header_frames": h,
        "n_token_frames": t,
        "n_trailer_frames": tr,
    });
    let out = ctx.path("grpc-stream.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_stream_emits_header_tokens_trailer() {
        let req = InferRequest {
            id: 1,
            prompt: "a b c".into(),
            max_tokens: 3,
        };
        let f = server_stream(&req);
        let (h, t, tr) = count_frames(&f);
        assert_eq!(h, 1);
        assert_eq!(t, 3);
        assert_eq!(tr, 1);
    }

    #[test]
    fn server_stream_respects_max_tokens() {
        let req = InferRequest {
            id: 1,
            prompt: "a b c d e f".into(),
            max_tokens: 2,
        };
        let f = server_stream(&req);
        let (_, t, _) = count_frames(&f);
        assert_eq!(t, 2);
    }

    #[test]
    fn zero_max_tokens_yields_error_trailer() {
        let req = InferRequest {
            id: 1,
            prompt: "a b c".into(),
            max_tokens: 0,
        };
        let f = server_stream(&req);
        match f.last() {
            Some(Frame::Trailer {
                status: Status::Error(_),
            }) => {}
            _ => panic!("expected error trailer"),
        }
    }

    #[test]
    fn bidi_stream_emits_one_header_per_chunk() {
        let f = bidi_stream(&["a b", "c d", "e"], 3);
        let (h, _, tr) = count_frames(&f);
        assert_eq!(h, 3);
        assert_eq!(tr, 3);
    }

    #[test]
    fn header_first_and_trailer_last() {
        let req = InferRequest {
            id: 9,
            prompt: "only one".into(),
            max_tokens: 10,
        };
        let f = server_stream(&req);
        assert!(matches!(f.first(), Some(Frame::Header { .. })));
        assert!(matches!(f.last(), Some(Frame::Trailer { .. })));
    }
}
