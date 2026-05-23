//! # apr ollama-chat-lint — Message Content Validation
//!
//! Each `messages[*].content` must be a non-empty string (per Ollama
//! API). Empty content from assistant means the model failed to generate
//! anything; from user means the request is malformed. Tool messages
//! must additionally include `tool_call_id`. This recipe builds the
//! per-message validator.
//!
//! Demonstrates the **OLLAMA-CHAT.6** recipe for PMAT-108 (apr ollama-chat-lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-C-04
//!
//! Run with: cargo run --example cli_ollama_chat_lint_message_content_validation
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum MessageVerdict {
    Ok,
    EmptyContent { role: String },
    UnknownRole { observed: String },
    ToolMissingCallId,
    SystemHasContent, // system message with empty content is OK; this case is for invalid edge — left as Ok
}

pub fn validate_message(m: &Message) -> MessageVerdict {
    if !["system", "user", "assistant", "tool"].contains(&m.role.as_str()) {
        return MessageVerdict::UnknownRole {
            observed: m.role.clone(),
        };
    }
    if m.content.trim().is_empty() {
        // Allow empty system message (no special instructions) but flag
        // empty user/assistant/tool content.
        if m.role != "system" {
            return MessageVerdict::EmptyContent {
                role: m.role.clone(),
            };
        }
    }
    if m.role == "tool" && m.tool_call_id.is_none() {
        return MessageVerdict::ToolMissingCallId;
    }
    MessageVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_chat_lint_message_content_validation")?;

    let cases = [
        (
            "user happy",
            Message {
                role: "user".into(),
                content: "hi".into(),
                tool_call_id: None,
            },
        ),
        (
            "assistant empty",
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_call_id: None,
            },
        ),
        (
            "system empty (ok)",
            Message {
                role: "system".into(),
                content: String::new(),
                tool_call_id: None,
            },
        ),
        (
            "tool no id",
            Message {
                role: "tool".into(),
                content: "result".into(),
                tool_call_id: None,
            },
        ),
        (
            "tool with id",
            Message {
                role: "tool".into(),
                content: "result".into(),
                tool_call_id: Some("call_123".into()),
            },
        ),
        (
            "unknown role",
            Message {
                role: "robot".into(),
                content: "hi".into(),
                tool_call_id: None,
            },
        ),
    ];
    for (label, m) in cases {
        println!("{label:>22}  →  {:?}", validate_message(&m));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn user_with_content_passes() {
        let m = Message {
            role: "user".into(),
            content: "hello".into(),
            tool_call_id: None,
        };
        assert_eq!(validate_message(&m), MessageVerdict::Ok);
    }

    #[test]
    fn empty_user_content_rejected() {
        let m = Message {
            role: "user".into(),
            content: String::new(),
            tool_call_id: None,
        };
        assert!(matches!(
            validate_message(&m),
            MessageVerdict::EmptyContent { .. }
        ));
    }

    #[test]
    fn empty_assistant_content_rejected() {
        let m = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_call_id: None,
        };
        assert!(matches!(
            validate_message(&m),
            MessageVerdict::EmptyContent { .. }
        ));
    }

    #[test]
    fn empty_system_content_allowed() {
        // System message with no instructions is a valid no-op.
        let m = Message {
            role: "system".into(),
            content: String::new(),
            tool_call_id: None,
        };
        assert_eq!(validate_message(&m), MessageVerdict::Ok);
    }

    #[test]
    fn whitespace_only_content_rejected() {
        // "   " is effectively empty.
        let m = Message {
            role: "user".into(),
            content: "   ".into(),
            tool_call_id: None,
        };
        assert!(matches!(
            validate_message(&m),
            MessageVerdict::EmptyContent { .. }
        ));
    }

    #[test]
    fn unknown_role_rejected() {
        let m = Message {
            role: "robot".into(),
            content: "hi".into(),
            tool_call_id: None,
        };
        assert!(matches!(
            validate_message(&m),
            MessageVerdict::UnknownRole { .. }
        ));
    }

    #[test]
    fn tool_without_call_id_rejected() {
        let m = Message {
            role: "tool".into(),
            content: "result".into(),
            tool_call_id: None,
        };
        assert_eq!(validate_message(&m), MessageVerdict::ToolMissingCallId);
    }

    #[test]
    fn tool_with_call_id_passes() {
        let m = Message {
            role: "tool".into(),
            content: "result".into(),
            tool_call_id: Some("call_abc".into()),
        };
        assert_eq!(validate_message(&m), MessageVerdict::Ok);
    }
}
