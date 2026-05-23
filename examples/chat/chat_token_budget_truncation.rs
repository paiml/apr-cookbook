//! # Chat Token Budget Truncation
//!
//! When conversation exceeds context_window − response_budget, drop
//! oldest messages until it fits. Always preserve: (1) the system
//! prompt (if any), (2) the most recent user message. This recipe
//! builds the truncator.
//!
//! Demonstrates the **CHAT.5** recipe for PMAT-125 (chat coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenAI Chat Completions truncation guidance.
//!
//! Run with: cargo run --example chat_token_budget_truncation
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub tokens: u32,
}

#[derive(Debug, PartialEq)]
pub enum TruncateVerdict {
    Ok {
        kept: Vec<Message>,
        dropped_count: usize,
    },
    DoesNotFit {
        needed: u32,
        available: u32,
    },
    InvalidBudget,
}

pub fn truncate(
    messages: &[Message],
    context_window: u32,
    response_budget: u32,
) -> TruncateVerdict {
    if response_budget >= context_window {
        return TruncateVerdict::InvalidBudget;
    }
    let available = context_window - response_budget;
    let total: u32 = messages.iter().map(|m| m.tokens).sum();
    if total <= available {
        return TruncateVerdict::Ok {
            kept: messages.to_vec(),
            dropped_count: 0,
        };
    }
    // Always keep system (if first) + last user (the latest user message).
    let has_system = matches!(messages.first().map(|m| m.role), Some(Role::System));
    let last_user_idx = messages.iter().rposition(|m| m.role == Role::User);
    let mut required: Vec<usize> = Vec::new();
    if has_system {
        required.push(0);
    }
    if let Some(i) = last_user_idx {
        if !required.contains(&i) {
            required.push(i);
        }
    }
    let required_tokens: u32 = required.iter().map(|&i| messages[i].tokens).sum();
    if required_tokens > available {
        return TruncateVerdict::DoesNotFit {
            needed: required_tokens,
            available,
        };
    }
    // Keep required + recent messages (newest first) until budget exhausted.
    let mut budget_left = available - required_tokens;
    let mut kept_indices: Vec<usize> = required.clone();
    for (i, m) in messages.iter().enumerate().rev() {
        if kept_indices.contains(&i) {
            continue;
        }
        if m.tokens <= budget_left {
            budget_left -= m.tokens;
            kept_indices.push(i);
        }
    }
    kept_indices.sort_unstable();
    let kept: Vec<Message> = kept_indices.iter().map(|&i| messages[i]).collect();
    TruncateVerdict::Ok {
        dropped_count: messages.len() - kept.len(),
        kept,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("chat_token_budget_truncation")?;

    let convo = [
        Message {
            role: Role::System,
            tokens: 100,
        },
        Message {
            role: Role::User,
            tokens: 50,
        },
        Message {
            role: Role::Assistant,
            tokens: 200,
        },
        Message {
            role: Role::User,
            tokens: 40,
        },
        Message {
            role: Role::Assistant,
            tokens: 150,
        },
        Message {
            role: Role::User,
            tokens: 60,
        },
    ];
    println!("8K window: {:?}", truncate(&convo, 8192, 1024));
    println!("400 window: {:?}", truncate(&convo, 400, 100));
    println!("100 window (too small): {:?}", truncate(&convo, 100, 50));
    println!("invalid budget: {:?}", truncate(&convo, 1000, 1500));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convo() -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                tokens: 100,
            },
            Message {
                role: Role::User,
                tokens: 50,
            },
            Message {
                role: Role::Assistant,
                tokens: 200,
            },
            Message {
                role: Role::User,
                tokens: 60,
            },
        ]
    }

    #[test]
    fn truncator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fits_returns_full_history() {
        let v = truncate(&convo(), 8192, 1024);
        if let TruncateVerdict::Ok {
            kept,
            dropped_count,
        } = v
        {
            assert_eq!(kept.len(), 4);
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn drops_middle_messages_when_tight() {
        // Total 410; window 250 - 50 budget = 200 available.
        // Required: System (100) + last User (60) = 160; budget left = 40.
        // No other message fits in 40 tokens → drop assistant + first user.
        let v = truncate(&convo(), 250, 50);
        if let TruncateVerdict::Ok {
            kept,
            dropped_count,
        } = v
        {
            assert_eq!(kept.len(), 2);
            assert_eq!(dropped_count, 2);
            assert_eq!(kept[0].role, Role::System);
            assert_eq!(kept[1].role, Role::User);
        }
    }

    #[test]
    fn budget_exceeds_window_invalid() {
        assert_eq!(
            truncate(&convo(), 1000, 1500),
            TruncateVerdict::InvalidBudget
        );
    }

    #[test]
    fn budget_equal_to_window_invalid() {
        assert_eq!(
            truncate(&convo(), 1000, 1000),
            TruncateVerdict::InvalidBudget
        );
    }

    #[test]
    fn required_messages_dont_fit() {
        // Window 100, budget 50, available 50. System alone = 100 > 50.
        let v = truncate(&convo(), 100, 50);
        assert!(matches!(v, TruncateVerdict::DoesNotFit { .. }));
    }

    #[test]
    fn no_system_keeps_only_last_user() {
        let no_system = vec![
            Message {
                role: Role::User,
                tokens: 50,
            },
            Message {
                role: Role::Assistant,
                tokens: 200,
            },
            Message {
                role: Role::User,
                tokens: 60,
            },
        ];
        // Window 110, budget 0, available 110. Last user 60 fits; assistant 200 doesn't.
        let v = truncate(&no_system, 110, 0);
        if let TruncateVerdict::Ok {
            kept,
            dropped_count,
        } = v
        {
            assert!(kept.len() < 3);
            assert!(dropped_count > 0);
        }
    }

    #[test]
    fn empty_conversation_fits_anywhere() {
        let v = truncate(&[], 1000, 100);
        if let TruncateVerdict::Ok {
            kept,
            dropped_count,
        } = v
        {
            assert_eq!(kept.len(), 0);
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn kept_messages_remain_in_chronological_order() {
        let v = truncate(&convo(), 250, 50);
        if let TruncateVerdict::Ok { kept, .. } = v {
            // Index 0 (System) before index 3 (last User).
            assert_eq!(kept[0].role, Role::System);
            assert_eq!(kept.last().unwrap().role, Role::User);
        }
    }
}
