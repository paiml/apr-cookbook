//! # Recipe: Multi-Format Chat Template Router
//!
//! **Category**: chat
//! **CLI Equivalent**: `apr chat` (auto-detect format from model name)
//! **APR Spec**: APR-021 (Chat Template Support)
//!
//! ## What this demonstrates
//!
//! A unified interface that auto-detects the correct chat template format
//! based on model name and applies the appropriate formatting. This mirrors
//! the `apr chat` CLI which selects the template automatically.
//!
//! ## Supported formats
//!
//! | Format  | Models                                   |
//! |---------|------------------------------------------|
//! | ChatML  | Qwen, Yi, OpenHermes, many fine-tunes    |
//! | LLaMA 2 | LLaMA-2-*-chat, CodeLlama-*-Instruct    |
//! | Mistral | Mistral-*-Instruct, Mixtral-*-Instruct   |
//! | Phi     | Phi-3-*, Phi-2-*                         |
//! | Alpaca  | Alpaca-*, Stanford Alpaca variants        |
//!
//! ## Sections
//! 1. Format detection from model names
//! 2. Side-by-side output comparison
//! 3. Token count comparison
//! 4. Determinism verification
//!
//! ## QA Checklist
//!
//! - [x] Compiles with `cargo build --example chat_multi_format`
//! - [x] Runs with `cargo run --example chat_multi_format`
//! - [x] Tests pass with `cargo test --example chat_multi_format`
//! - [x] No unsafe code
//! - [x] No unwrap on user data
//! - [x] Clippy clean
//!
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

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemplateFormat {
    ChatML,
    Llama2,
    Mistral,
    Phi,
    Alpaca,
}

impl std::fmt::Display for TemplateFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChatML => write!(f, "ChatML"),
            Self::Llama2 => write!(f, "LLaMA 2"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Alpaca => write!(f, "Alpaca"),
        }
    }
}

/// Detect the appropriate chat template format from a model name.
///
/// Uses case-insensitive substring matching against known model families.
/// Falls back to ChatML as the most widely compatible format.
fn detect_format(model_name: &str) -> TemplateFormat {
    let lower = model_name.to_lowercase();

    if lower.contains("mistral") || lower.contains("mixtral") {
        TemplateFormat::Mistral
    } else if lower.contains("llama-2") || lower.contains("llama2") || lower.contains("codellama") {
        TemplateFormat::Llama2
    } else if lower.contains("phi-") || lower.contains("phi2") || lower.contains("phi3") {
        TemplateFormat::Phi
    } else if lower.contains("alpaca") {
        TemplateFormat::Alpaca
    } else {
        // ChatML is the default for Qwen, Yi, OpenHermes, and most fine-tunes
        TemplateFormat::ChatML
    }
}

/// Format messages in ChatML format.
fn format_chatml(messages: &[ChatMessage], add_gen: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    if add_gen {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Extract the first system message from a conversation, returning
/// its content and the remaining non-system-leading messages.
fn extract_system_prefix(messages: &[ChatMessage]) -> (Option<&str>, Vec<&ChatMessage>) {
    let mut system: Option<&str> = None;
    let mut conv: Vec<&ChatMessage> = Vec::new();
    for msg in messages {
        if msg.role == "system" && system.is_none() && conv.is_empty() {
            system = Some(&msg.content);
        } else {
            conv.push(msg);
        }
    }
    (system, conv)
}

/// Format a single LLaMA 2 turn (one user message and optional assistant response).
/// Returns the number of messages consumed (1 or 2).
fn format_llama2_turn(
    out: &mut String,
    conv: &[&ChatMessage],
    index: usize,
    system: Option<&str>,
    add_gen: bool,
) -> usize {
    out.push_str("<s>[INST] ");
    if index == 0 {
        if let Some(sys) = system {
            out.push_str(&format!("<<SYS>>\n{sys}\n<</SYS>>\n\n"));
        }
    }
    out.push_str(&conv[index].content);
    out.push_str(" [/INST]");
    if index + 1 < conv.len() && conv[index + 1].role == "assistant" {
        out.push_str(&format!(" {} </s>", conv[index + 1].content));
        2
    } else {
        if add_gen {
            out.push(' ');
        }
        1
    }
}

/// Format messages in LLaMA 2 format.
fn format_llama2(messages: &[ChatMessage], add_gen: bool) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let (system, conv) = extract_system_prefix(messages);
    let mut out = String::new();
    let mut i = 0;
    while i < conv.len() {
        i += format_llama2_turn(&mut out, &conv, i, system, add_gen);
    }
    out
}

/// Format a single Mistral turn (one `[INST]...[/INST]` block).
/// Returns the number of messages consumed (1 or 2), or 0 if
/// the current message is not a user message (skip it).
fn format_mistral_turn(
    out: &mut String,
    conv: &[&ChatMessage],
    index: usize,
    sys_prefix: Option<&str>,
    add_gen: bool,
) -> usize {
    if conv[index].role != "user" {
        return 1;
    }
    out.push_str("[INST] ");
    if index == 0 {
        if let Some(prefix) = sys_prefix {
            out.push_str(prefix);
            out.push_str("\n\n");
        }
    }
    out.push_str(&conv[index].content);
    out.push_str(" [/INST]");
    if index + 1 < conv.len() && conv[index + 1].role == "assistant" {
        out.push_str(&format!(" {}</s>", conv[index + 1].content));
        2
    } else {
        if add_gen {
            out.push(' ');
        }
        1
    }
}

/// Format messages in Mistral format.
fn format_mistral(messages: &[ChatMessage], add_gen: bool) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let (system, conv) = extract_system_prefix(messages);
    let sys_prefix = system.filter(|s| !s.is_empty());
    let mut out = String::from("<s>");
    let mut i = 0;
    while i < conv.len() {
        i += format_mistral_turn(&mut out, &conv, i, sys_prefix, add_gen);
    }
    out
}

/// Format messages in Phi format.
///
/// Phi uses a tag-based format: `<|user|>\ncontent<|end|>\n<|assistant|>\n`
fn format_phi(messages: &[ChatMessage], add_gen: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }
    if add_gen {
        out.push_str("<|assistant|>\n");
    }
    out
}

/// Format messages in Alpaca format.
///
/// Alpaca uses a prompt-style format:
/// ```text
/// ### Instruction:
/// {system}
///
/// ### Input:
/// {user}
///
/// ### Response:
/// {assistant}
/// ```
fn format_alpaca(messages: &[ChatMessage], add_gen: bool) -> String {
    let mut out = String::new();
    let mut system_content: Option<&str> = None;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                system_content = Some(&msg.content);
            }
            "user" => {
                if let Some(sys) = system_content.take() {
                    out.push_str(&format!("### Instruction:\n{sys}\n\n"));
                }
                out.push_str(&format!("### Input:\n{}\n\n", msg.content));
            }
            "assistant" => {
                out.push_str(&format!("### Response:\n{}\n\n", msg.content));
            }
            _ => {}
        }
    }
    if add_gen {
        out.push_str("### Response:\n");
    }
    out
}

/// Format messages using the detected template format.
fn format_messages(
    format: TemplateFormat,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> String {
    match format {
        TemplateFormat::ChatML => format_chatml(messages, add_generation_prompt),
        TemplateFormat::Llama2 => format_llama2(messages, add_generation_prompt),
        TemplateFormat::Mistral => format_mistral(messages, add_generation_prompt),
        TemplateFormat::Phi => format_phi(messages, add_generation_prompt),
        TemplateFormat::Alpaca => format_alpaca(messages, add_generation_prompt),
    }
}

/// Estimate token count for a formatted string (rough: ~4 chars per token).
fn estimate_tokens(formatted: &str) -> usize {
    formatted.len().div_ceil(4)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("chat_multi_format")?;

    // --- Section 1: Format detection ---
    println!("=== Format Detection ===");

    let test_models = vec![
        ("mistral-7b-instruct-v0.2", TemplateFormat::Mistral),
        ("llama-2-13b-chat", TemplateFormat::Llama2),
        ("phi-3-mini-4k-instruct", TemplateFormat::Phi),
        ("qwen2-7b-instruct", TemplateFormat::ChatML),
        ("alpaca-7b", TemplateFormat::Alpaca),
        ("yi-34b-chat", TemplateFormat::ChatML),
        ("codellama-34b-instruct", TemplateFormat::Llama2),
        ("mixtral-8x7b-instruct", TemplateFormat::Mistral),
    ];

    for (model, expected) in &test_models {
        let detected = detect_format(model);
        println!("{model:<40} -> {detected}");
        assert_eq!(
            detected, *expected,
            "Format mismatch for {model}: got {detected}, expected {expected}"
        );
    }

    ctx.record_metric("models_tested", test_models.len() as i64);

    // --- Section 2: Side-by-side output comparison ---
    println!("\n=== Side-by-Side Comparison ===");

    let messages = vec![
        ChatMessage::new("system", "You are a helpful assistant."),
        ChatMessage::new("user", "What is APR?"),
    ];

    let formats = [
        TemplateFormat::ChatML,
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::Phi,
        TemplateFormat::Alpaca,
    ];

    for fmt in &formats {
        let output = format_messages(*fmt, &messages, true);
        println!("--- {fmt} ({} bytes) ---", output.len());
        println!("{output}");
    }

    // --- Section 3: Token count comparison ---
    println!("\n=== Token Count Comparison ===");

    let messages = vec![
        ChatMessage::new("system", "You are a concise ML assistant."),
        ChatMessage::new("user", "Explain quantization in one sentence."),
        ChatMessage::new(
            "assistant",
            "Quantization reduces model precision to save memory and speed up inference.",
        ),
        ChatMessage::new("user", "What about FP16 vs INT8?"),
    ];

    println!("Format          | Bytes | Est. Tokens");
    println!("----------------|-------|------------");

    for fmt in &formats {
        let output = format_messages(*fmt, &messages, true);
        let tokens = estimate_tokens(&output);
        println!("{fmt:<15} | {:<5} | {tokens}", output.len());
        ctx.record_metric(&format!("{fmt}_bytes"), output.len() as i64);
    }

    // --- Section 4: Determinism verification ---
    println!("\n=== Determinism Check ===");

    let messages = vec![ChatMessage::new("user", "Test message")];

    for fmt in &formats {
        let a = format_messages(*fmt, &messages, true);
        let b = format_messages(*fmt, &messages, true);
        assert_eq!(a, b, "{fmt} format must be deterministic");
    }
    println!("All formats produce deterministic output.");

    ctx.record_metric("formats_verified", formats.len() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_mistral() {
        assert_eq!(detect_format("mistral-7b"), TemplateFormat::Mistral);
        assert_eq!(
            detect_format("Mistral-7B-Instruct-v0.2"),
            TemplateFormat::Mistral
        );
        assert_eq!(detect_format("mixtral-8x7b"), TemplateFormat::Mistral);
    }

    #[test]
    fn test_detect_llama2() {
        assert_eq!(detect_format("llama-2-13b-chat"), TemplateFormat::Llama2);
        assert_eq!(detect_format("Llama-2-70B-chat-hf"), TemplateFormat::Llama2);
        assert_eq!(
            detect_format("codellama-34b-instruct"),
            TemplateFormat::Llama2
        );
    }

    #[test]
    fn test_detect_phi() {
        assert_eq!(detect_format("phi-3-mini-4k-instruct"), TemplateFormat::Phi);
        assert_eq!(detect_format("Phi-3-medium"), TemplateFormat::Phi);
    }

    #[test]
    fn test_detect_alpaca_and_chatml_default() {
        assert_eq!(detect_format("alpaca-7b"), TemplateFormat::Alpaca);
        assert_eq!(detect_format("qwen2-7b"), TemplateFormat::ChatML);
        assert_eq!(detect_format("yi-34b-chat"), TemplateFormat::ChatML);
        assert_eq!(detect_format("unknown-model"), TemplateFormat::ChatML);
    }

    #[test]
    fn test_all_formats_produce_output() {
        let messages = vec![ChatMessage::new("user", "Hello")];
        let formats = [
            TemplateFormat::ChatML,
            TemplateFormat::Llama2,
            TemplateFormat::Mistral,
            TemplateFormat::Phi,
            TemplateFormat::Alpaca,
        ];
        for fmt in &formats {
            let output = format_messages(*fmt, &messages, true);
            assert!(!output.is_empty(), "{fmt} must produce non-empty output");
        }
    }

    #[test]
    fn test_all_formats_deterministic() {
        let messages = vec![
            ChatMessage::new("system", "Sys"),
            ChatMessage::new("user", "Usr"),
        ];
        let formats = [
            TemplateFormat::ChatML,
            TemplateFormat::Llama2,
            TemplateFormat::Mistral,
            TemplateFormat::Phi,
            TemplateFormat::Alpaca,
        ];
        for fmt in &formats {
            let a = format_messages(*fmt, &messages, true);
            let b = format_messages(*fmt, &messages, true);
            assert_eq!(a, b, "{fmt} must be deterministic");
        }
    }

    #[test]
    fn test_chatml_format_correct() {
        let messages = vec![ChatMessage::new("user", "Hi")];
        let out = format_chatml(&messages, true);
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_phi_format_correct() {
        let messages = vec![ChatMessage::new("user", "Hi")];
        let out = format_phi(&messages, true);
        assert!(out.contains("<|user|>\nHi<|end|>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_alpaca_format_correct() {
        let messages = vec![
            ChatMessage::new("system", "Be helpful."),
            ChatMessage::new("user", "Hi"),
        ];
        let out = format_alpaca(&messages, true);
        assert!(out.contains("### Instruction:\nBe helpful."));
        assert!(out.contains("### Input:\nHi"));
        assert!(out.ends_with("### Response:\n"));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_empty_messages_all_formats() {
        let formats = [
            TemplateFormat::ChatML,
            TemplateFormat::Llama2,
            TemplateFormat::Mistral,
            TemplateFormat::Phi,
            TemplateFormat::Alpaca,
        ];
        for fmt in &formats {
            let output = format_messages(*fmt, &[], false);
            assert!(output.is_empty(), "{fmt} with empty messages must be empty");
        }
    }

    #[test]
    fn test_template_format_display() {
        assert_eq!(format!("{}", TemplateFormat::ChatML), "ChatML");
        assert_eq!(format!("{}", TemplateFormat::Llama2), "LLaMA 2");
        assert_eq!(format!("{}", TemplateFormat::Mistral), "Mistral");
        assert_eq!(format!("{}", TemplateFormat::Phi), "Phi");
        assert_eq!(format!("{}", TemplateFormat::Alpaca), "Alpaca");
    }

    #[test]
    fn test_format_messages_dispatches_correctly() {
        let messages = vec![ChatMessage::new("user", "Test")];
        let chatml = format_messages(TemplateFormat::ChatML, &messages, false);
        let llama2 = format_messages(TemplateFormat::Llama2, &messages, false);
        // Different formats must produce different output
        assert_ne!(chatml, llama2, "ChatML and LLaMA 2 should differ");
    }
}
