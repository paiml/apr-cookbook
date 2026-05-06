//! # apr ollama-chat-lint — Role State Machine
//!
//! Ollama /api/chat conversations follow a role state machine:
//! `system` (optional, ≤1, must be first) → `user` → `assistant` →
//! `user` → `assistant` → … (alternating). Tool messages may interleave
//! between assistant and user. This recipe builds the validator and
//! enforces the contract.
//!
//! Demonstrates the **OLLAMA-CHAT.5** recipe for PMAT-108 (apr ollama-chat-lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-C-04 + Ollama API role conventions
//!
//! Run with: cargo run --example cli_ollama_chat_lint_role_state_machine
//!
//! Added by PMAT-108 (expand-cookbooks followup).

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
pub enum RoleVerdict {
    Ok,
    SystemNotFirst { at_index: usize },
    MultipleSystem { count: usize },
    InvalidStart { observed: Role },
    ConsecutiveSameRole { at_index: usize, role: Role },
    EmptyConversation,
}

pub fn validate_role_sequence(roles: &[Role]) -> RoleVerdict {
    if roles.is_empty() {
        return RoleVerdict::EmptyConversation;
    }
    // Count system roles and check position.
    let system_count = roles.iter().filter(|r| **r == Role::System).count();
    if system_count > 1 {
        return RoleVerdict::MultipleSystem {
            count: system_count,
        };
    }
    if system_count == 1 && roles[0] != Role::System {
        let pos = roles.iter().position(|r| *r == Role::System).unwrap();
        return RoleVerdict::SystemNotFirst { at_index: pos };
    }
    // After optional system, sequence must start with user.
    let first_non_system = if roles[0] == Role::System {
        if roles.len() < 2 {
            return RoleVerdict::Ok;
        }
        roles[1]
    } else {
        roles[0]
    };
    if first_non_system != Role::User {
        return RoleVerdict::InvalidStart {
            observed: first_non_system,
        };
    }
    // Check no two consecutive non-tool messages of the same role.
    for (i, w) in roles.windows(2).enumerate() {
        if w[0] == w[1] && w[0] != Role::Tool {
            return RoleVerdict::ConsecutiveSameRole {
                at_index: i + 1,
                role: w[0],
            };
        }
    }
    RoleVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_chat_lint_role_state_machine")?;

    let cases = [
        (
            "happy",
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::User,
                Role::Assistant,
            ],
        ),
        ("no system", vec![Role::User, Role::Assistant]),
        (
            "system not first",
            vec![Role::User, Role::System, Role::Assistant],
        ),
        ("two system", vec![Role::System, Role::System, Role::User]),
        ("starts with assistant", vec![Role::Assistant, Role::User]),
        (
            "two consecutive user",
            vec![Role::User, Role::User, Role::Assistant],
        ),
        (
            "tool interleaved (ok)",
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant],
        ),
        ("empty", vec![]),
    ];
    for (label, r) in cases {
        println!("{label:>22}  →  {:?}", validate_role_sequence(&r));
    }
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
    fn empty_conversation_rejected() {
        assert_eq!(validate_role_sequence(&[]), RoleVerdict::EmptyConversation);
    }

    #[test]
    fn happy_sequence_passes() {
        let r = vec![Role::System, Role::User, Role::Assistant];
        assert_eq!(validate_role_sequence(&r), RoleVerdict::Ok);
    }

    #[test]
    fn no_system_passes() {
        let r = vec![Role::User, Role::Assistant];
        assert_eq!(validate_role_sequence(&r), RoleVerdict::Ok);
    }

    #[test]
    fn multiple_system_rejected() {
        let r = vec![Role::System, Role::System, Role::User];
        assert!(matches!(
            validate_role_sequence(&r),
            RoleVerdict::MultipleSystem { count: 2 }
        ));
    }

    #[test]
    fn system_not_first_rejected() {
        let r = vec![Role::User, Role::System, Role::Assistant];
        assert!(matches!(
            validate_role_sequence(&r),
            RoleVerdict::SystemNotFirst { .. }
        ));
    }

    #[test]
    fn starting_with_assistant_rejected() {
        let r = vec![Role::Assistant, Role::User];
        assert!(matches!(
            validate_role_sequence(&r),
            RoleVerdict::InvalidStart { .. }
        ));
    }

    #[test]
    fn consecutive_same_role_rejected() {
        let r = vec![Role::User, Role::User, Role::Assistant];
        assert!(matches!(
            validate_role_sequence(&r),
            RoleVerdict::ConsecutiveSameRole { .. }
        ));
    }

    #[test]
    fn consecutive_tool_messages_allowed() {
        // Tool messages can interleave / repeat — they're not part of the
        // user/assistant alternation.
        let r = vec![
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Tool,
            Role::Assistant,
        ];
        assert_eq!(validate_role_sequence(&r), RoleVerdict::Ok);
    }
}
