#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFormat {
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

// Detect the appropriate chat template format from a model name.
//
// Uses case-insensitive substring matching against known model families.
/// Falls back to ChatML as the most widely compatible format.
pub fn detect_format(model_name: &str) -> TemplateFormat {
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
pub fn format_chatml(messages: &[ChatMessage], add_gen: bool) -> String {
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

// Extract the first system message from a conversation, returning
/// its content and the remaining non-system-leading messages.
pub fn extract_system_prefix(messages: &[ChatMessage]) -> (Option<&str>, Vec<&ChatMessage>) {
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

// Format a single LLaMA 2 turn (one user message and optional assistant response).
/// Returns the number of messages consumed (1 or 2).
pub fn format_llama2_turn(
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
pub fn format_llama2(messages: &[ChatMessage], add_gen: bool) -> String {
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

// Format a single Mistral turn (one `[INST]...[/INST]` block).
// Returns the number of messages consumed (1 or 2), or 0 if
/// the current message is not a user message (skip it).
pub fn format_mistral_turn(
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
pub fn format_mistral(messages: &[ChatMessage], add_gen: bool) -> String {
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

// Format messages in Phi format.
//
/// Phi uses a tag-based format: `<|user|>\ncontent<|end|>\n<|assistant|>\n`
pub fn format_phi(messages: &[ChatMessage], add_gen: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }
    if add_gen {
        out.push_str("<|assistant|>\n");
    }
    out
}

// Format messages in Alpaca format.
//
// Alpaca uses a prompt-style format:
// ```text
// ### Instruction:
// {system}
//
// ### Input:
// {user}
//
// ### Response:
// {assistant}
/// ```
pub fn format_alpaca(messages: &[ChatMessage], add_gen: bool) -> String {
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
pub fn format_messages(
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
pub fn estimate_tokens(formatted: &str) -> usize {
    formatted.len().div_ceil(4)
}
