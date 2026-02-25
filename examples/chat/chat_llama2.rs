//! # Recipe: LLaMA 2 Chat Template Formatting
//!
//! **Category**: chat
//! **CLI Equivalent**: `apr chat --format llama2`
//! **APR Spec**: APR-021 (Chat Template Support)
//!
//! ## What this demonstrates
//!
//! LLaMA 2 uses a unique chat format with `[INST]` / `[/INST]` delimiters
//! and a `<<SYS>>` block for system prompts. This example implements the
//! full LLaMA 2 chat template specification, including multi-turn handling
//! where system prompts are only included in the first turn.
//!
//! ## Format specification
//!
//! ```text
//! <s>[INST] <<SYS>>
//! system message
//! <</SYS>>
//!
//! user message [/INST] assistant response </s>
//! <s>[INST] next user message [/INST]
//! ```
//!
//! ## Sections
//! 1. Basic user message
//! 2. System prompt placement
//! 3. Multi-turn conversation
//! 4. Comparison with ChatML
//!
//! ## QA Checklist
//!
//! - [x] Compiles with `cargo build --example chat_llama2`
//! - [x] Runs with `cargo run --example chat_llama2`
//! - [x] Tests pass with `cargo test --example chat_llama2`
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

/// LLaMA 2 special tokens and delimiters.
const BOS: &str = "<s>";
const EOS: &str = "</s>";
const INST_START: &str = "[INST]";
const INST_END: &str = "[/INST]";
const SYS_START: &str = "<<SYS>>";
const SYS_END: &str = "<</SYS>>";

/// Format a system prompt in LLaMA 2 style.
///
/// Wraps the system message in `<<SYS>>` delimiters with proper newlines.
fn format_system_block(system_content: &str) -> String {
    format!("{SYS_START}\n{system_content}\n{SYS_END}\n\n")
}

/// Format a sequence of chat messages in LLaMA 2 chat format.
///
/// Rules:
/// - System prompt (if present) is embedded in the first `[INST]` block.
/// - User/assistant messages alternate in `[INST]`/`[/INST]` pairs.
/// - Each complete turn is wrapped with `<s>` and `</s>`.
/// - The last user message gets no `</s>` if `add_generation_prompt` is true.
fn format_llama2(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    let mut system_prompt: Option<&str> = None;
    let mut conversation: Vec<&ChatMessage> = Vec::new();

    // Separate system prompt from conversation messages
    for msg in messages {
        if msg.role == "system" && system_prompt.is_none() {
            system_prompt = Some(&msg.content);
        } else {
            conversation.push(msg);
        }
    }

    // Process user/assistant pairs
    let mut i = 0;
    while i < conversation.len() {
        let user_msg = &conversation[i];
        assert_eq!(
            user_msg.role, "user",
            "Expected user message at position {i}"
        );

        output.push_str(BOS);
        output.push_str(INST_START);
        output.push(' ');

        // Include system prompt in the first turn only
        if i == 0 {
            if let Some(sys) = system_prompt {
                output.push_str(&format_system_block(sys));
            }
        }

        output.push_str(&user_msg.content);
        output.push(' ');
        output.push_str(INST_END);

        // Check for assistant response
        if i + 1 < conversation.len() && conversation[i + 1].role == "assistant" {
            output.push(' ');
            output.push_str(&conversation[i + 1].content);
            output.push(' ');
            output.push_str(EOS);
            i += 2;
        } else {
            // Last message is user-only
            if add_generation_prompt {
                output.push(' ');
            }
            i += 1;
        }
    }

    output
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("chat_llama2")?;

    // --- Section 1: Basic user message ---
    println!("=== Basic Format ===");

    let messages = vec![ChatMessage::new("user", "What is the APR format?")];
    let formatted = format_llama2(&messages, true);
    println!("Basic user message:\n{formatted}");

    assert!(formatted.contains(INST_START), "Must contain [INST]");
    assert!(formatted.contains(INST_END), "Must contain [/INST]");
    assert!(formatted.starts_with(BOS), "Must start with BOS token");

    ctx.record_metric("basic_msg_bytes", formatted.len() as i64);

    // --- Section 2: With system prompt ---
    println!("\n=== System Prompt ===");

    let messages = vec![
        ChatMessage::new("system", "You are an expert in ML model formats."),
        ChatMessage::new("user", "Explain APR compression."),
    ];
    let formatted = format_llama2(&messages, true);
    println!("With system prompt:\n{formatted}");

    assert!(formatted.contains(SYS_START), "Must contain <<SYS>>");
    assert!(formatted.contains(SYS_END), "Must contain <</SYS>>");
    assert!(
        formatted.find(SYS_START).expect("SYS_START present")
            < formatted.find("Explain APR").expect("user msg present"),
        "System prompt must come before user message"
    );

    // --- Section 3: Multi-turn conversation ---
    println!("\n=== Multi-Turn Conversation ===");

    let messages = vec![
        ChatMessage::new("system", "Be concise."),
        ChatMessage::new("user", "What is quantization?"),
        ChatMessage::new("assistant", "Reducing model precision to save memory."),
        ChatMessage::new("user", "What precisions does APR support?"),
    ];
    let formatted = format_llama2(&messages, true);
    println!("Multi-turn:\n{formatted}");

    let inst_count = formatted.matches(INST_START).count();
    println!("Number of [INST] blocks: {inst_count}");
    assert_eq!(inst_count, 2, "Two user turns = two [INST] blocks");

    ctx.record_metric("multi_turn_inst_blocks", inst_count as i64);

    // System prompt only in the first turn
    let first_inst = formatted.find(INST_START).expect("first INST");
    let second_inst_start = first_inst + INST_START.len();
    let second_inst = formatted[second_inst_start..].find(INST_START);
    if let Some(offset) = second_inst {
        let second_block = &formatted[second_inst_start + offset..];
        assert!(
            !second_block.contains(SYS_START),
            "System prompt must NOT appear in second turn"
        );
    }

    // --- Section 4: Comparison with ChatML ---
    println!("\n=== Format Comparison ===");

    let messages = vec![
        ChatMessage::new("system", "You are helpful."),
        ChatMessage::new("user", "Hello!"),
    ];
    let llama2_out = format_llama2(&messages, true);

    // Approximate ChatML for comparison
    let chatml_out = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                      <|im_start|>user\nHello!<|im_end|>\n\
                      <|im_start|>assistant\n";

    println!("LLaMA 2 ({} bytes):\n{llama2_out}", llama2_out.len());
    println!("ChatML  ({} bytes):\n{chatml_out}", chatml_out.len());
    println!("LLaMA 2 nests system inside [INST]; ChatML uses separate role blocks.");

    ctx.record_metric("llama2_bytes", llama2_out.len() as i64);
    ctx.record_metric("chatml_bytes", chatml_out.len() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_user_message() {
        let messages = vec![ChatMessage::new("user", "Hello")];
        let formatted = format_llama2(&messages, false);
        assert_eq!(formatted, "<s>[INST] Hello [/INST]");
    }

    #[test]
    fn test_user_with_generation_prompt() {
        let messages = vec![ChatMessage::new("user", "Hello")];
        let formatted = format_llama2(&messages, true);
        assert!(formatted.contains("[/INST] "));
    }

    #[test]
    fn test_system_prompt_placement() {
        let messages = vec![
            ChatMessage::new("system", "Be helpful."),
            ChatMessage::new("user", "Hi"),
        ];
        let formatted = format_llama2(&messages, false);
        assert!(formatted.contains("<<SYS>>\nBe helpful.\n<</SYS>>"));
    }

    #[test]
    fn test_system_prompt_before_user() {
        let messages = vec![
            ChatMessage::new("system", "System msg"),
            ChatMessage::new("user", "User msg"),
        ];
        let formatted = format_llama2(&messages, false);
        let sys_pos = formatted.find("System msg").expect("system present");
        let usr_pos = formatted.find("User msg").expect("user present");
        assert!(sys_pos < usr_pos, "System must come before user");
    }

    #[test]
    fn test_user_assistant_pair() {
        let messages = vec![
            ChatMessage::new("user", "What is Rust?"),
            ChatMessage::new("assistant", "A systems language."),
        ];
        let formatted = format_llama2(&messages, false);
        assert!(formatted.contains("[/INST] A systems language. </s>"));
    }

    #[test]
    fn test_multi_turn_structure() {
        let messages = vec![
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
        ];
        let formatted = format_llama2(&messages, false);
        assert_eq!(formatted.matches("[INST]").count(), 2);
        assert_eq!(formatted.matches("[/INST]").count(), 2);
        assert_eq!(
            formatted.matches("</s>").count(),
            1,
            "Only completed turn gets EOS"
        );
    }

    #[test]
    fn test_system_only_in_first_turn() {
        let messages = vec![
            ChatMessage::new("system", "Be brief."),
            ChatMessage::new("user", "Q1"),
            ChatMessage::new("assistant", "A1"),
            ChatMessage::new("user", "Q2"),
        ];
        let formatted = format_llama2(&messages, false);
        // Find second [INST] block and verify no <<SYS>> in it
        let first_end = formatted.find("[/INST]").expect("first end");
        let rest = &formatted[first_end..];
        assert!(
            !rest.contains("<<SYS>>"),
            "System prompt must not appear in later turns"
        );
    }

    #[test]
    fn test_bos_token_present() {
        let messages = vec![ChatMessage::new("user", "Hi")];
        let formatted = format_llama2(&messages, false);
        assert!(formatted.starts_with("<s>"), "Must begin with BOS token");
    }

    #[test]
    fn test_eos_after_assistant() {
        let messages = vec![
            ChatMessage::new("user", "Hi"),
            ChatMessage::new("assistant", "Hello!"),
        ];
        let formatted = format_llama2(&messages, false);
        assert!(
            formatted.ends_with("</s>"),
            "Must end with EOS after assistant"
        );
    }

    #[test]
    fn test_empty_messages() {
        let formatted = format_llama2(&[], false);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_format_deterministic() {
        let messages = vec![
            ChatMessage::new("system", "Sys"),
            ChatMessage::new("user", "Usr"),
        ];
        let a = format_llama2(&messages, true);
        let b = format_llama2(&messages, true);
        assert_eq!(a, b);
    }

    #[test]
    fn test_multi_turn_with_system() {
        let messages = vec![
            ChatMessage::new("system", "You are an AI."),
            ChatMessage::new("user", "Hello"),
            ChatMessage::new("assistant", "Hi!"),
            ChatMessage::new("user", "Bye"),
            ChatMessage::new("assistant", "Goodbye!"),
        ];
        let formatted = format_llama2(&messages, false);
        assert_eq!(formatted.matches("<s>").count(), 2, "Two turns = two BOS");
        assert_eq!(
            formatted.matches("</s>").count(),
            2,
            "Two complete turns = two EOS"
        );
    }

    #[test]
    fn test_format_system_block() {
        let block = format_system_block("Test system");
        assert_eq!(block, "<<SYS>>\nTest system\n<</SYS>>\n\n");
    }
}
