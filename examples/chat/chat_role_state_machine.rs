//! # Chat Role Transition State Machine
//!
//! Chat sessions follow strict role ordering: optional `system` first,
//! then alternating `user` ↔ `assistant`. The optional `tool` role
//! can interject after `assistant` (tool call) and is followed by
//! `assistant` (tool result interpretation). This recipe codifies the
//! transitions.
//!
//! Demonstrates the **CHAT.4** recipe for PMAT-125 (chat coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenAI ChatML spec; Anthropic Messages API roles.
//!
//! Run with: cargo run --example chat_role_state_machine
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok,
    SystemNotFirst,
    NoUserAfterSystem,
    DoubleUser,
    DoubleAssistant,
    ToolWithoutPriorAssistant,
    NonAssistantAfterTool,
    EmptyConversation,
}

pub fn validate(roles: &[Role]) -> TransitionVerdict {
    if roles.is_empty() {
        return TransitionVerdict::EmptyConversation;
    }
    // System can only be at index 0.
    if roles.iter().skip(1).any(|r| *r == Role::System) {
        return TransitionVerdict::SystemNotFirst;
    }
    // First message: System or User.
    if roles[0] == Role::System {
        if roles.len() < 2 || roles[1] != Role::User {
            return TransitionVerdict::NoUserAfterSystem;
        }
    } else if roles[0] != Role::User {
        return TransitionVerdict::NoUserAfterSystem;
    }
    // Walk transitions.
    for w in roles.windows(2) {
        match (w[0], w[1]) {
            (Role::User, Role::User) => return TransitionVerdict::DoubleUser,
            (Role::Assistant, Role::Assistant) => return TransitionVerdict::DoubleAssistant,
            (Role::Tool, next) if next != Role::Assistant => {
                return TransitionVerdict::NonAssistantAfterTool;
            }
            _ => {}
        }
    }
    // Tool always preceded by Assistant.
    for (i, r) in roles.iter().enumerate() {
        if *r == Role::Tool && (i == 0 || roles[i - 1] != Role::Assistant) {
            return TransitionVerdict::ToolWithoutPriorAssistant;
        }
    }
    TransitionVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("chat_role_state_machine")?;

    let valid = [
        Role::System,
        Role::User,
        Role::Assistant,
        Role::User,
        Role::Assistant,
    ];
    let with_tool = [Role::User, Role::Assistant, Role::Tool, Role::Assistant];
    let bad_double = [Role::User, Role::Assistant, Role::Assistant];
    let bad_system = [Role::User, Role::System];

    println!("valid:        {:?}", validate(&valid));
    println!("with_tool:    {:?}", validate(&with_tool));
    println!("bad_double:   {:?}", validate(&bad_double));
    println!("bad_system:   {:?}", validate(&bad_system));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_alternating_passes() {
        let v = validate(&[
            Role::System,
            Role::User,
            Role::Assistant,
            Role::User,
            Role::Assistant,
        ]);
        assert_eq!(v, TransitionVerdict::Ok);
    }

    #[test]
    fn user_first_without_system_passes() {
        let v = validate(&[Role::User, Role::Assistant]);
        assert_eq!(v, TransitionVerdict::Ok);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[]), TransitionVerdict::EmptyConversation);
    }

    #[test]
    fn system_not_first_rejected() {
        assert_eq!(
            validate(&[Role::User, Role::System]),
            TransitionVerdict::SystemNotFirst
        );
    }

    #[test]
    fn system_alone_or_followed_by_assistant_rejected() {
        // System without subsequent User is invalid.
        let v = validate(&[Role::System]);
        assert_eq!(v, TransitionVerdict::NoUserAfterSystem);
        let v2 = validate(&[Role::System, Role::Assistant]);
        assert_eq!(v2, TransitionVerdict::NoUserAfterSystem);
    }

    #[test]
    fn double_user_rejected() {
        assert_eq!(
            validate(&[Role::User, Role::User]),
            TransitionVerdict::DoubleUser
        );
    }

    #[test]
    fn double_assistant_rejected() {
        assert_eq!(
            validate(&[Role::User, Role::Assistant, Role::Assistant]),
            TransitionVerdict::DoubleAssistant
        );
    }

    #[test]
    fn tool_after_assistant_passes() {
        let v = validate(&[Role::User, Role::Assistant, Role::Tool, Role::Assistant]);
        assert_eq!(v, TransitionVerdict::Ok);
    }

    #[test]
    fn tool_without_prior_assistant_rejected() {
        let v = validate(&[Role::User, Role::Tool]);
        assert_eq!(v, TransitionVerdict::ToolWithoutPriorAssistant);
    }

    #[test]
    fn user_after_tool_rejected() {
        // Tool result must go to Assistant, not back to User.
        let v = validate(&[Role::User, Role::Assistant, Role::Tool, Role::User]);
        assert_eq!(v, TransitionVerdict::NonAssistantAfterTool);
    }
}
