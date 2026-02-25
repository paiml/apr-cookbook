//! # Recipe: Mistral Chat Template Formatting
//!
//! **Category**: chat
//! **CLI Equivalent**: `apr chat --format mistral`
//! **APR Spec**: APR-021 (Chat Template Support)
//!
//! ## What this demonstrates
//!
//! Mistral uses a minimal chat format with `[INST]` / `[/INST]` delimiters
//! but notably does NOT support a dedicated system prompt role. System
//! instructions must be prepended to the first user message. This example
//! implements the Mistral Instruct template and compares it with LLaMA 2.
//!
//! ## Format specification
//!
//! ```text
//! <s>[INST] user message [/INST] assistant response</s>[INST] next user [/INST]
//! ```
//!
//! ## Sections
//! 1. Basic user message
//! 2. Multi-turn conversation
//! 3. System prompt handling (no native support)
//! 4. Comparison with LLaMA 2
//! 5. Extended multi-turn
//!
//! ## QA Checklist
//!
//! - [x] Compiles with `cargo build --example chat_mistral`
//! - [x] Runs with `cargo run --example chat_mistral`
//! - [x] Tests pass with `cargo test --example chat_mistral`
//! - [x] No unsafe code
//! - [x] No unwrap on user data
//! - [x] Clippy clean

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

/// Mistral special tokens.
const BOS: &str = "<s>";
const EOS: &str = "</s>";
const INST_START: &str = "[INST]";
const INST_END: &str = "[/INST]";

/// Format a sequence of chat messages in Mistral Instruct format.
///
/// Key differences from LLaMA 2:
/// - No `<<SYS>>` block: system messages are prepended to the first user message.
/// - BOS token only at the very beginning (not per-turn).
/// - EOS token after each assistant response.
/// - No space padding around assistant response (tighter format).
fn format_mistral(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    let mut system_prefix = String::new();
    let mut conversation: Vec<&ChatMessage> = Vec::new();

    // Extract system messages and prepend to first user message
    for msg in messages {
        if msg.role == "system" && conversation.is_empty() && system_prefix.is_empty() {
            system_prefix = msg.content.clone();
        } else {
            conversation.push(msg);
        }
    }

    // BOS token at the start
    output.push_str(BOS);

    let mut i = 0;
    while i < conversation.len() {
        let msg = &conversation[i];

        if msg.role == "user" {
            output.push_str(INST_START);
            output.push(' ');

            // Prepend system instructions to first user message
            if i == 0 && !system_prefix.is_empty() {
                output.push_str(&system_prefix);
                output.push_str("\n\n");
            }

            output.push_str(&msg.content);
            output.push(' ');
            output.push_str(INST_END);

            // Check for following assistant message
            if i + 1 < conversation.len() && conversation[i + 1].role == "assistant" {
                output.push(' ');
                output.push_str(&conversation[i + 1].content);
                output.push_str(EOS);
                i += 2;
            } else {
                if add_generation_prompt {
                    output.push(' ');
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    output
}

/// Check whether the Mistral format contains a system prompt block.
///
/// Returns false because Mistral does not have a native system role.
fn has_native_system_support() -> bool {
    false
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("chat_mistral")?;

    // --- Section 1: Basic user message ---
    println!("=== Basic Format ===");

    let messages = vec![ChatMessage::new("user", "What is the APR format?")];
    let formatted = format_mistral(&messages, true);
    println!("Basic user message:\n{formatted}");

    assert!(formatted.starts_with(BOS), "Must start with BOS");
    assert!(formatted.contains(INST_START), "Must contain [INST]");
    assert!(formatted.contains(INST_END), "Must contain [/INST]");

    ctx.record_metric("basic_msg_bytes", formatted.len() as i64);

    // --- Section 2: Multi-turn conversation ---
    println!("\n=== Multi-Turn Conversation ===");

    let messages = vec![
        ChatMessage::new("user", "What is quantization?"),
        ChatMessage::new(
            "assistant",
            "Reducing numerical precision of model weights.",
        ),
        ChatMessage::new("user", "What about FP16?"),
    ];
    let formatted = format_mistral(&messages, true);
    println!("Multi-turn:\n{formatted}");

    let inst_count = formatted.matches(INST_START).count();
    println!("Number of [INST] blocks: {inst_count}");
    assert_eq!(inst_count, 2, "Two user messages = two [INST] blocks");

    // Only one BOS at the start (not per-turn like LLaMA 2)
    assert_eq!(
        formatted.matches(BOS).count(),
        1,
        "Mistral uses single BOS token"
    );

    ctx.record_metric("multi_turn_inst_blocks", inst_count as i64);

    // --- Section 3: No native system prompt ---
    println!("\n=== System Prompt Handling ===");

    println!("Native system support: {}", has_native_system_support());
    println!("Mistral prepends system instructions to the first user message.");

    let messages = vec![
        ChatMessage::new("system", "You are an ML expert."),
        ChatMessage::new("user", "Explain LZ4 compression."),
    ];
    let formatted = format_mistral(&messages, true);
    println!("System as prefix:\n{formatted}");

    assert!(
        !formatted.contains("<<SYS>>"),
        "Mistral must NOT use <<SYS>> block"
    );
    // System content should be present but inline with user message
    assert!(
        formatted.contains("You are an ML expert."),
        "System content must be included"
    );
    assert!(
        formatted.contains("Explain LZ4 compression."),
        "User content must be included"
    );

    ctx.record_string_metric("system_handling", "prepend_to_user");

    // --- Section 4: Comparison with LLaMA 2 ---
    println!("\n=== Format Comparison ===");

    let messages = vec![
        ChatMessage::new("user", "Hello!"),
        ChatMessage::new("assistant", "Hi!"),
        ChatMessage::new("user", "How are you?"),
    ];
    let mistral_out = format_mistral(&messages, true);

    println!("Mistral ({} bytes):\n{mistral_out}", mistral_out.len());
    println!("Key differences from LLaMA 2:");
    println!("  1. Single BOS token at start (not per-turn)");
    println!("  2. No <<SYS>> block");
    println!("  3. EOS directly after assistant (no trailing space)");
    println!("  4. Tighter format = fewer tokens");

    ctx.record_metric("comparison_bytes", mistral_out.len() as i64);

    // --- Section 5: Extended multi-turn ---
    println!("\n=== Extended Conversation ===");

    let messages = vec![
        ChatMessage::new("user", "Q1"),
        ChatMessage::new("assistant", "A1"),
        ChatMessage::new("user", "Q2"),
        ChatMessage::new("assistant", "A2"),
        ChatMessage::new("user", "Q3"),
    ];
    let formatted = format_mistral(&messages, true);
    println!("5-message conversation:\n{formatted}");

    assert_eq!(formatted.matches(INST_START).count(), 3, "Three user turns");
    assert_eq!(
        formatted.matches(EOS).count(),
        2,
        "Two completed assistant turns"
    );

    ctx.record_metric("extended_turn_count", 3);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_user_message() {
        let messages = vec![ChatMessage::new("user", "Hello")];
        let formatted = format_mistral(&messages, false);
        assert_eq!(formatted, "<s>[INST] Hello [/INST]");
    }

    #[test]
    fn test_user_assistant_pair() {
        let messages = vec![
            ChatMessage::new("user", "Hi"),
            ChatMessage::new("assistant", "Hello!"),
        ];
        let formatted = format_mistral(&messages, false);
        assert_eq!(formatted, "<s>[INST] Hi [/INST] Hello!</s>");
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
        ];
        let formatted = format_mistral(&messages, false);
        assert!(formatted.contains("A1</s>"));
        assert!(formatted.contains("[INST] Q2 [/INST]"));
    }

    #[test]
    fn test_single_bos_token() {
        let messages = vec![
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
        ];
        let formatted = format_mistral(&messages, false);
        assert_eq!(formatted.matches("<s>").count(), 1, "Only one BOS token");
    }

    #[test]
    fn test_no_native_system_support() {
        assert!(!has_native_system_support());
    }

    #[test]
    fn test_system_prepended_to_user() {
        let messages = vec![
            ChatMessage::new("system", "Be helpful."),
            ChatMessage::new("user", "Hi"),
        ];
        let formatted = format_mistral(&messages, false);
        // System should appear before user content within the same [INST] block
        let inst_content_start = formatted.find("[INST] ").expect("INST present") + 7;
        let inst_content_end = formatted.find(" [/INST]").expect("INST end");
        let inst_content = &formatted[inst_content_start..inst_content_end];
        assert!(
            inst_content.starts_with("Be helpful."),
            "System prefix first"
        );
        assert!(inst_content.contains("Hi"), "User content follows");
    }

    #[test]
    fn test_no_sys_delimiters() {
        let messages = vec![
            ChatMessage::new("system", "Sys"),
            ChatMessage::new("user", "Usr"),
        ];
        let formatted = format_mistral(&messages, false);
        assert!(!formatted.contains("<<SYS>>"));
        assert!(!formatted.contains("<</SYS>>"));
    }

    #[test]
    fn test_generation_prompt() {
        let messages = vec![ChatMessage::new("user", "Test")];
        let with = format_mistral(&messages, true);
        let without = format_mistral(&messages, false);
        assert!(with.len() >= without.len());
    }

    #[test]
    fn test_empty_messages() {
        let formatted = format_mistral(&[], false);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_format_deterministic() {
        let messages = vec![
            ChatMessage::new("user", "Q"),
            ChatMessage::new("assistant", "A"),
        ];
        let a = format_mistral(&messages, true);
        let b = format_mistral(&messages, true);
        assert_eq!(a, b);
    }

    #[test]
    fn test_eos_after_each_assistant() {
        let messages = vec![
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
            ChatMessage::new("assistant", "A2"),
        ];
        let formatted = format_mistral(&messages, false);
        assert_eq!(formatted.matches("</s>").count(), 2);
    }

    #[test]
    fn test_extended_conversation_structure() {
        let messages = vec![
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
            ChatMessage::new("assistant", "A2"),
            ChatMessage::new("user", "Q3"),
        ];
        let formatted = format_mistral(&messages, false);
        assert_eq!(formatted.matches("[INST]").count(), 3);
        assert_eq!(formatted.matches("[/INST]").count(), 3);
    }
}
