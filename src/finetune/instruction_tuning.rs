//! Tier 3.1 Instruction tuning — shared helper for 5 recipes.
//!
//! Models the *observable* invariants of common SFT instruction-tuning data
//! formats and chat templates so each recipe falsifier is a tight,
//! deterministic check rather than a stochastic eval-uplift claim.
//!
//! - Alpaca: token-level loss-masking (system+instruction tokens masked).
//! - ShareGPT: multi-turn role-token round-trip (turns parse cleanly).
//! - OpenAssistant: tree-format flattening preserves leaf assistant turns.
//! - Chat templates: ChatML/Llama2/Mistral/Phi token-exact round-trip.
//! - System-prompt: prefix masked from loss but visible on forward pass.

#![allow(clippy::needless_range_loop)]

/// Alpaca-style row.
#[derive(Debug, Clone, PartialEq)]
pub struct AlpacaRow {
    pub instruction: String,
    pub input: String,
    pub output: String,
}

/// Build a flat token sequence + per-token loss mask for an Alpaca row.
/// Tokens for instruction/input are masked (mask=false); only `output`
/// tokens contribute to loss (mask=true).
#[must_use]
pub fn alpaca_flatten(row: &AlpacaRow) -> (Vec<String>, Vec<bool>) {
    let mut toks = Vec::new();
    let mut mask = Vec::new();
    let prefix = format!(
        "Instruction: {} Input: {} Output:",
        row.instruction, row.input
    );
    for tok in prefix.split_whitespace() {
        toks.push(tok.to_string());
        mask.push(false);
    }
    for tok in row.output.split_whitespace() {
        toks.push(tok.to_string());
        mask.push(true);
    }
    (toks, mask)
}

/// ShareGPT multi-turn row.
#[derive(Debug, Clone, PartialEq)]
pub struct ShareGPTTurn {
    pub role: String, // "user" or "assistant"
    pub content: String,
}

/// Parse a ShareGPT-style flat string of `[user] X [assistant] Y [user] Z`
/// back into structured turns. Returns parsed turns (or empty Vec on bad input).
#[must_use]
pub fn sharegpt_parse(flat: &str) -> Vec<ShareGPTTurn> {
    let mut turns = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_buf = String::new();
    for tok in flat.split_whitespace() {
        if tok == "[user]" || tok == "[assistant]" || tok == "[system]" {
            if let Some(role) = current_role.take() {
                let content = current_buf.trim().to_string();
                if !content.is_empty() {
                    turns.push(ShareGPTTurn { role, content });
                }
            }
            current_role = Some(tok.trim_matches(|c| c == '[' || c == ']').to_string());
            current_buf.clear();
        } else if current_role.is_some() {
            if !current_buf.is_empty() {
                current_buf.push(' ');
            }
            current_buf.push_str(tok);
        }
    }
    if let Some(role) = current_role {
        let content = current_buf.trim().to_string();
        if !content.is_empty() {
            turns.push(ShareGPTTurn { role, content });
        }
    }
    turns
}

/// Render structured turns back to flat ShareGPT format.
#[must_use]
pub fn sharegpt_render(turns: &[ShareGPTTurn]) -> String {
    turns
        .iter()
        .map(|t| format!("[{}] {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join(" ")
}

/// OpenAssistant tree node.
#[derive(Debug, Clone, PartialEq)]
pub struct OAMessage {
    pub role: String,
    pub text: String,
    pub children: Vec<OAMessage>,
}

/// Flatten an OA tree into a list of (role, text) pairs by depth-first
/// traversal taking the longest path. Returns flattened-leaf path.
#[must_use]
pub fn oa_flatten_longest_path(root: &OAMessage) -> Vec<(String, String)> {
    let mut best: Vec<(String, String)> = Vec::new();
    let mut path: Vec<(String, String)> = Vec::new();
    oa_dfs(root, &mut path, &mut best);
    best
}

fn oa_dfs(node: &OAMessage, path: &mut Vec<(String, String)>, best: &mut Vec<(String, String)>) {
    path.push((node.role.clone(), node.text.clone()));
    if node.children.is_empty() {
        if path.len() > best.len() {
            *best = path.clone();
        }
    } else {
        for child in &node.children {
            oa_dfs(child, path, best);
        }
    }
    path.pop();
}

/// Apply chat template to messages. Supports ChatML, Llama2, Mistral, Phi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    ChatML,
    Llama2,
    Mistral,
    Phi,
}

#[must_use]
pub fn render_chat(template: ChatTemplate, messages: &[(String, String)]) -> String {
    match template {
        ChatTemplate::ChatML => messages
            .iter()
            .map(|(r, t)| format!("<|im_start|>{r}\n{t}<|im_end|>"))
            .collect::<Vec<_>>()
            .join("\n"),
        ChatTemplate::Llama2 => messages
            .iter()
            .map(|(r, t)| match r.as_str() {
                "system" => format!("<<SYS>>\n{t}\n<</SYS>>"),
                "user" => format!("[INST] {t} [/INST]"),
                _ => t.clone(),
            })
            .collect::<Vec<_>>()
            .join(" "),
        ChatTemplate::Mistral => messages
            .iter()
            .map(|(r, t)| match r.as_str() {
                "user" => format!("[INST] {t} [/INST]"),
                _ => t.clone(),
            })
            .collect::<Vec<_>>()
            .join(" "),
        ChatTemplate::Phi => messages
            .iter()
            .map(|(r, t)| format!("<|{r}|>\n{t}<|end|>"))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Extract messages from a rendered chat string for the given template.
/// Returns parsed (role, text) pairs.
#[must_use]
pub fn parse_chat(template: ChatTemplate, rendered: &str) -> Vec<(String, String)> {
    match template {
        ChatTemplate::ChatML => {
            let mut out = Vec::new();
            let parts = rendered.split("<|im_start|>");
            for p in parts.skip(1) {
                if let Some((role, rest)) = p.split_once('\n') {
                    if let Some(end) = rest.find("<|im_end|>") {
                        out.push((role.trim().to_string(), rest[..end].trim().to_string()));
                    }
                }
            }
            out
        }
        ChatTemplate::Phi => {
            let mut out = Vec::new();
            let mut cursor = 0;
            while cursor < rendered.len() {
                let after_cursor = &rendered[cursor..];
                let role_start = match after_cursor.find("<|") {
                    Some(p) => cursor + p + 2,
                    None => break,
                };
                let role_end = match rendered[role_start..].find("|>") {
                    Some(p) => role_start + p,
                    None => break,
                };
                let role = rendered[role_start..role_end].to_string();
                let body_start = role_end + 2;
                let body_end = match rendered[body_start..].find("<|end|>") {
                    Some(p) => body_start + p,
                    None => break,
                };
                if role != "end" {
                    out.push((role, rendered[body_start..body_end].trim().to_string()));
                }
                cursor = body_end + "<|end|>".len();
            }
            out
        }
        // Llama2/Mistral round-trips are lossy — recover instruction blocks only.
        ChatTemplate::Llama2 | ChatTemplate::Mistral => {
            let mut out = Vec::new();
            let mut cursor = 0;
            while cursor < rendered.len() {
                if let Some(start) = rendered[cursor..].find("[INST]") {
                    let abs_start = cursor + start + "[INST]".len();
                    if let Some(end) = rendered[abs_start..].find("[/INST]") {
                        let content = rendered[abs_start..abs_start + end].trim().to_string();
                        out.push(("user".to_string(), content));
                        cursor = abs_start + end + "[/INST]".len();
                        continue;
                    }
                }
                break;
            }
            out
        }
    }
}

/// System-prompt loss masking: tokens belonging to the system prompt are
/// masked out of the loss but visible in the forward pass. Returns
/// (concat_tokens, loss_mask) where loss_mask=false for system, true for
/// the user/assistant turns that follow.
#[must_use]
pub fn system_prompt_mask(system: &str, body: &str) -> (Vec<String>, Vec<bool>) {
    let mut toks = Vec::new();
    let mut mask = Vec::new();
    for tok in system.split_whitespace() {
        toks.push(tok.to_string());
        mask.push(false);
    }
    for tok in body.split_whitespace() {
        toks.push(tok.to_string());
        mask.push(true);
    }
    (toks, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpaca_loss_mask_zero_on_prefix() {
        let row = AlpacaRow {
            instruction: "Translate to French".into(),
            input: "Hello".into(),
            output: "Bonjour".into(),
        };
        let (toks, mask) = alpaca_flatten(&row);
        // No prefix token (Instruction:/Input:/Output:) should be in the loss.
        let last_unmasked = mask.iter().rposition(|&v| !v).unwrap();
        let first_masked = mask.iter().position(|&v| v).unwrap();
        assert!(last_unmasked < first_masked);
        // The output "Bonjour" is the only loss-bearing token.
        assert_eq!(toks[first_masked], "Bonjour");
    }

    #[test]
    fn sharegpt_round_trip() {
        let turns = vec![
            ShareGPTTurn {
                role: "user".into(),
                content: "what is 2+2?".into(),
            },
            ShareGPTTurn {
                role: "assistant".into(),
                content: "4".into(),
            },
            ShareGPTTurn {
                role: "user".into(),
                content: "and 3+3?".into(),
            },
            ShareGPTTurn {
                role: "assistant".into(),
                content: "6".into(),
            },
        ];
        let flat = sharegpt_render(&turns);
        let parsed = sharegpt_parse(&flat);
        assert_eq!(parsed, turns);
    }

    #[test]
    fn oa_longest_path_picked() {
        let leaf_a = OAMessage {
            role: "assistant".into(),
            text: "answer A".into(),
            children: vec![],
        };
        let leaf_b = OAMessage {
            role: "assistant".into(),
            text: "answer B".into(),
            children: vec![OAMessage {
                role: "user".into(),
                text: "follow-up".into(),
                children: vec![OAMessage {
                    role: "assistant".into(),
                    text: "answer B2".into(),
                    children: vec![],
                }],
            }],
        };
        let root = OAMessage {
            role: "user".into(),
            text: "question".into(),
            children: vec![leaf_a, leaf_b],
        };
        let path = oa_flatten_longest_path(&root);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0].1, "question");
        assert_eq!(path.last().unwrap().1, "answer B2");
    }

    #[test]
    fn chatml_round_trip() {
        let messages = vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi".to_string()),
        ];
        let rendered = render_chat(ChatTemplate::ChatML, &messages);
        let parsed = parse_chat(ChatTemplate::ChatML, &rendered);
        assert_eq!(parsed, messages);
    }

    #[test]
    fn phi_round_trip() {
        let messages = vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi".to_string()),
        ];
        let rendered = render_chat(ChatTemplate::Phi, &messages);
        let parsed = parse_chat(ChatTemplate::Phi, &rendered);
        assert_eq!(parsed, messages);
    }

    #[test]
    fn system_prompt_mask_zeros_system_only() {
        let (_, mask) = system_prompt_mask("You are helpful.", "What is 2+2 plus 1?");
        // System: 3 tokens (You / are / helpful.), body: 5 tokens.
        assert_eq!(mask.iter().filter(|&&v| !v).count(), 3);
        assert_eq!(mask.iter().filter(|&&v| v).count(), 5);
    }
}
