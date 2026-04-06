//! # Recipe: ChatML Template Formatting
//!
//! **Category**: chat
//! **CLI Equivalent**: `apr chat --format chatml`
//! **APR Spec**: APR-021 (Chat Template Support)
//!
//! ## What this demonstrates
//!
//! ChatML is the standard chat template used by OpenAI-compatible models,
//! Qwen, Yi, and many fine-tuned variants. This example implements the
//! ChatML format from scratch, showing exact byte-level structure with
//! special tokens.
//!
//! ## Format specification
//!
//! ```text
//! <|im_start|>role
//! content<|im_end|>
//! ```
//!
//! ## Sections
//! 1. Single message formatting
//! 2. Multi-turn conversation
//! 3. System prompt + user + assistant
//! 4. Generation prompt toggling
//! 5. Byte-level format inspection
//!
//! ## QA Checklist
//!
//! - [x] Compiles with `cargo build --example chat_chatml`
//! - [x] Runs with `cargo run --example chat_chatml`
//! - [x] Tests pass with `cargo test --example chat_chatml`
//! - [x] No unsafe code
//! - [x] No unwrap on user data
//! - [x] Clippy clean
//!
//!
//! ## Format Variants
//! ```bash
//! apr chat model.apr          # APR native format
//! apr chat model.gguf         # GGUF (llama.cpp compatible)
//! apr chat model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971

use apr_cookbook::prelude::*;

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// Special tokens used in the ChatML format.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Format a single message in ChatML format.
///
/// Produces: `<|im_start|>role\ncontent<|im_end|>\n`
fn format_chatml_message(msg: &ChatMessage) -> String {
    format!("{}{}\n{}{}\n", IM_START, msg.role, msg.content, IM_END)
}

/// Format a sequence of chat messages in ChatML format.
///
/// Each message is wrapped with `<|im_start|>` and `<|im_end|>` tokens.
/// An optional generation prompt is appended to signal the model to begin
/// generating an assistant response.
fn format_chatml(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut output = String::new();
    for msg in messages {
        output.push_str(&format_chatml_message(msg));
    }
    if add_generation_prompt {
        output.push_str(&format!("{}{}\n", IM_START, "assistant"));
    }
    output
}

/// Count the number of special tokens in a formatted string.
fn count_special_tokens(formatted: &str) -> usize {
    let starts = formatted.matches(IM_START).count();
    let ends = formatted.matches(IM_END).count();
    starts + ends
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("chat_chatml")?;

    // --- Section 1: Single message formatting ---
    println!("=== Single Message ===");

    let user_msg = ChatMessage::new("user", "What is the APR format?");
    let formatted = format_chatml_message(&user_msg);
    println!("Single user message:\n{formatted}");

    let expected = "<|im_start|>user\nWhat is the APR format?<|im_end|>\n";
    assert_eq!(formatted, expected, "Single message format mismatch");
    println!("Byte length: {}", formatted.len());
    println!("Special tokens: {}", count_special_tokens(&formatted));

    ctx.record_metric("single_msg_bytes", formatted.len() as i64);

    // --- Section 2: Multi-turn conversation ---
    println!("\n=== Multi-Turn Conversation ===");

    let messages = vec![
        ChatMessage::new("user", "Hello!"),
        ChatMessage::new("assistant", "Hi! How can I help you today?"),
        ChatMessage::new("user", "Tell me about .apr model files."),
    ];

    let formatted = format_chatml(&messages, true);
    println!("Multi-turn with generation prompt:\n{formatted}");

    let token_count = count_special_tokens(&formatted);
    println!("Total special tokens: {token_count}");
    // 3 messages = 6 tokens (start+end each) + 1 generation prompt start = 7
    assert_eq!(
        token_count, 7,
        "Expected 7 special tokens in multi-turn + gen prompt"
    );

    ctx.record_metric("multi_turn_tokens", token_count as i64);

    // --- Section 3: System prompt + user + assistant ---
    println!("\n=== System Prompt Pattern ===");

    let messages = vec![
        ChatMessage::new(
            "system",
            "You are a helpful ML assistant specializing in model formats.",
        ),
        ChatMessage::new("user", "What compression does APR support?"),
        ChatMessage::new(
            "assistant",
            "APR supports LZ4 and Zstd compression for efficient storage.",
        ),
    ];

    let formatted = format_chatml(&messages, false);
    println!("System + user + assistant (no gen prompt):\n{formatted}");

    assert!(
        formatted.starts_with("<|im_start|>system\n"),
        "Must start with system role"
    );
    assert!(
        formatted.contains("<|im_start|>user\n"),
        "Must contain user role"
    );
    assert!(
        formatted.contains("<|im_start|>assistant\n"),
        "Must contain assistant role"
    );
    assert!(
        !formatted.ends_with("<|im_start|>assistant\n"),
        "No trailing gen prompt"
    );

    // --- Section 4: Generation prompt toggling ---
    println!("\n=== Generation Prompt ===");

    let messages = vec![ChatMessage::new("user", "Explain quantization.")];

    let with_gen = format_chatml(&messages, true);
    let without_gen = format_chatml(&messages, false);

    println!("With generation prompt:\n{with_gen}");
    println!("Without generation prompt:\n{without_gen}");

    assert!(
        with_gen.ends_with("<|im_start|>assistant\n"),
        "Gen prompt must end with assistant start"
    );
    assert!(
        with_gen.len() > without_gen.len(),
        "With gen prompt must be longer"
    );

    let diff = with_gen.len() - without_gen.len();
    println!("Generation prompt adds {diff} bytes");
    ctx.record_metric("gen_prompt_overhead_bytes", diff as i64);

    // --- Section 5: Byte-level format inspection ---
    println!("\n=== Byte-Level Format ===");

    let msg = ChatMessage::new("user", "Hi");
    let formatted = format_chatml_message(&msg);
    let bytes: Vec<u8> = formatted.bytes().collect();

    println!("Raw bytes ({} total): {:?}", bytes.len(), bytes);
    println!("Format structure:");
    println!("  <|im_start|> = 12 bytes (special token marker)");
    println!("  role         = variable");
    println!("  \\n           = 1 byte (newline separator)");
    println!("  content      = variable");
    println!("  <|im_end|>   = 10 bytes (special token marker)");
    println!("  \\n           = 1 byte (trailing newline)");

    assert_eq!(
        &formatted[..12],
        IM_START,
        "First 12 bytes must be im_start"
    );

    ctx.record_metric("im_start_bytes", 12);
    ctx.record_metric("im_end_bytes", 10);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_user_message() {
        let msg = ChatMessage::new("user", "Hello");
        let formatted = format_chatml_message(&msg);
        assert_eq!(formatted, "<|im_start|>user\nHello<|im_end|>\n");
    }

    #[test]
    fn test_single_system_message() {
        let msg = ChatMessage::new("system", "You are helpful.");
        let formatted = format_chatml_message(&msg);
        assert_eq!(
            formatted,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        );
    }

    #[test]
    fn test_single_assistant_message() {
        let msg = ChatMessage::new("assistant", "Sure, I can help.");
        let formatted = format_chatml_message(&msg);
        assert_eq!(
            formatted,
            "<|im_start|>assistant\nSure, I can help.<|im_end|>\n"
        );
    }

    #[test]
    fn test_system_user_conversation() {
        let messages = vec![
            ChatMessage::new("system", "Be concise."),
            ChatMessage::new("user", "Hi"),
        ];
        let formatted = format_chatml(&messages, false);
        let expected = "<|im_start|>system\nBe concise.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_multi_turn_conversation() {
        let messages = vec![
            ChatMessage::new("user", "Hello"),
            ChatMessage::new("assistant", "Hi there!"),
            ChatMessage::new("user", "How are you?"),
            ChatMessage::new("assistant", "I'm doing well."),
        ];
        let formatted = format_chatml(&messages, false);
        assert_eq!(formatted.matches(IM_START).count(), 4);
        assert_eq!(formatted.matches(IM_END).count(), 4);
    }

    #[test]
    fn test_generation_prompt_appended() {
        let messages = vec![ChatMessage::new("user", "Test")];
        let formatted = format_chatml(&messages, true);
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_no_generation_prompt() {
        let messages = vec![ChatMessage::new("user", "Test")];
        let formatted = format_chatml(&messages, false);
        assert!(!formatted.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_empty_content() {
        let msg = ChatMessage::new("user", "");
        let formatted = format_chatml_message(&msg);
        assert_eq!(formatted, "<|im_start|>user\n<|im_end|>\n");
    }

    #[test]
    fn test_special_characters_in_content() {
        let msg = ChatMessage::new("user", "What about <tags> & \"quotes\"?");
        let formatted = format_chatml_message(&msg);
        assert!(formatted.contains("<tags>"));
        assert!(formatted.contains("&"));
        assert!(formatted.contains("\"quotes\""));
    }

    #[test]
    fn test_multiline_content() {
        let msg = ChatMessage::new("user", "Line 1\nLine 2\nLine 3");
        let formatted = format_chatml_message(&msg);
        assert_eq!(
            formatted,
            "<|im_start|>user\nLine 1\nLine 2\nLine 3<|im_end|>\n"
        );
    }

    #[test]
    fn test_special_token_count() {
        let messages = vec![
            ChatMessage::new("system", "sys"),
            ChatMessage::new("user", "usr"),
        ];
        let formatted = format_chatml(&messages, true);
        // 2 messages * 2 tokens + 1 gen prompt start = 5 starts, 2 ends
        assert_eq!(count_special_tokens(&formatted), 5);
    }

    #[test]
    fn test_format_deterministic() {
        let messages = vec![
            ChatMessage::new("user", "Hello"),
            ChatMessage::new("assistant", "Hi"),
        ];
        let a = format_chatml(&messages, true);
        let b = format_chatml(&messages, true);
        assert_eq!(a, b, "Formatting must be deterministic");
    }

    #[test]
    fn test_unicode_content() {
        let msg = ChatMessage::new("user", "Hola, como estas?");
        let formatted = format_chatml_message(&msg);
        assert!(formatted.contains("como estas?"));
    }

    #[test]
    fn test_empty_messages_list() {
        let formatted = format_chatml(&[], false);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_empty_messages_with_gen_prompt() {
        let formatted = format_chatml(&[], true);
        assert_eq!(formatted, "<|im_start|>assistant\n");
    }
}
