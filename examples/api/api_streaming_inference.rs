//! # Recipe: Streaming Model Inference
//!
//! **Category**: API Integration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Stream model outputs token-by-token (simulated).
//!
//! ## Run Command
//! ```bash
//! cargo run --example api_streaming_inference
//! ```

use apr_cookbook::prelude::*;
use std::collections::VecDeque;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("api_streaming_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Streaming model inference (simulated)");
    println!();

    // Create streaming inference session
    let mut session = StreamingSession::new(StreamConfig {
        max_tokens: 20,
        temperature: 0.7,
        buffer_size: 4,
    });

    // Input prompt
    let prompt = "The quick brown fox";
    println!("Prompt: {}", prompt);
    println!();

    // Initialize stream
    session.start(prompt);
    ctx.record_metric("prompt_tokens", prompt.split_whitespace().count() as i64);

    // Stream tokens
    println!("Streaming output:");
    print!("  ");

    let mut total_tokens = 0;
    while let Some(token) = session.next_token() {
        print!("{} ", token);
        total_tokens += 1;
    }
    println!();

    ctx.record_metric("output_tokens", total_tokens);
    ctx.record_metric("total_chunks", session.chunk_count() as i64);

    println!();
    println!("Statistics:");
    println!("  Total tokens: {}", total_tokens);
    println!("  Chunks sent: {}", session.chunk_count());
    println!(
        "  Avg tokens/chunk: {:.1}",
        total_tokens as f64 / session.chunk_count() as f64
    );

    // Save streaming log
    let log_path = ctx.path("stream_log.txt");
    session.save_log(&log_path)?;
    println!();
    println!("Stream log saved to: {:?}", log_path);

    Ok(())
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StreamConfig {
    max_tokens: usize,
    temperature: f32,
    buffer_size: usize,
}

#[derive(Debug)]
struct StreamingSession {
    config: StreamConfig,
    buffer: VecDeque<String>,
    tokens_generated: usize,
    chunks_sent: usize,
    seed: u64,
    log: Vec<String>,
}

impl StreamingSession {
    fn new(config: StreamConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            tokens_generated: 0,
            chunks_sent: 0,
            seed: 42,
            log: Vec::new(),
        }
    }

    fn start(&mut self, prompt: &str) {
        self.log.push(format!("START: {}", prompt));
        // Pre-fill buffer with mock tokens
        self.refill_buffer();
    }

    fn next_token(&mut self) -> Option<String> {
        if self.tokens_generated >= self.config.max_tokens {
            return None;
        }

        // Refill buffer if needed
        if self.buffer.is_empty() {
            self.refill_buffer();
            self.chunks_sent += 1;
        }

        let token = self.buffer.pop_front()?;
        self.tokens_generated += 1;
        self.log
            .push(format!("TOKEN[{}]: {}", self.tokens_generated, token));

        Some(token)
    }

    fn refill_buffer(&mut self) {
        // Deterministic mock token generation
        let tokens = [
            "jumps", "over", "the", "lazy", "dog", "and", "runs", "through", "the", "forest",
            "with", "great", "speed", "while", "hunting", "for", "food", "in", "the", "wild",
        ];

        for i in 0..self.config.buffer_size {
            let idx = (self.seed as usize + self.tokens_generated + i) % tokens.len();
            self.buffer.push_back(tokens[idx].to_string());
        }
    }

    fn chunk_count(&self) -> usize {
        self.chunks_sent.max(1)
    }

    fn save_log(&self, path: &std::path::Path) -> Result<()> {
        let content = self.log.join("\n");
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_session_creation() {
        let session = StreamingSession::new(StreamConfig {
            max_tokens: 10,
            temperature: 1.0,
            buffer_size: 4,
        });

        assert_eq!(session.tokens_generated, 0);
        assert!(session.buffer.is_empty());
    }

    #[test]
    fn test_token_generation() {
        let mut session = StreamingSession::new(StreamConfig {
            max_tokens: 5,
            temperature: 1.0,
            buffer_size: 2,
        });

        session.start("test");

        let mut tokens = Vec::new();
        while let Some(token) = session.next_token() {
            tokens.push(token);
        }

        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_deterministic_output() {
        let config = StreamConfig {
            max_tokens: 10,
            temperature: 1.0,
            buffer_size: 4,
        };

        let mut session1 = StreamingSession::new(config.clone());
        let mut session2 = StreamingSession::new(config);

        session1.start("test");
        session2.start("test");

        let tokens1: Vec<_> = std::iter::from_fn(|| session1.next_token()).collect();
        let tokens2: Vec<_> = std::iter::from_fn(|| session2.next_token()).collect();

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_max_tokens_limit() {
        let mut session = StreamingSession::new(StreamConfig {
            max_tokens: 3,
            temperature: 1.0,
            buffer_size: 10,
        });

        session.start("test");

        let count = std::iter::from_fn(|| session.next_token()).count();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_log_save() {
        let ctx = RecipeContext::new("test_stream_log").unwrap();
        let path = ctx.path("log.txt");

        let mut session = StreamingSession::new(StreamConfig {
            max_tokens: 2,
            temperature: 1.0,
            buffer_size: 2,
        });

        session.start("hello");
        while session.next_token().is_some() {}

        session.save_log(&path).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_respects_max_tokens(max_tokens in 1usize..50) {
            let mut session = StreamingSession::new(StreamConfig {
                max_tokens,
                temperature: 1.0,
                buffer_size: 4,
            });

            session.start("test");
            let count = std::iter::from_fn(|| session.next_token()).count();

            prop_assert_eq!(count, max_tokens);
        }

        #[test]
        fn prop_tokens_not_empty(max_tokens in 1usize..20, buffer_size in 1usize..10) {
            let mut session = StreamingSession::new(StreamConfig {
                max_tokens,
                temperature: 1.0,
                buffer_size,
            });

            session.start("test");

            while let Some(token) = session.next_token() {
                prop_assert!(!token.is_empty());
            }
        }
    }
}
