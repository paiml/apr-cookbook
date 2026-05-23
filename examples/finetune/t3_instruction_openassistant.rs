//! # Tier 3.1 — Instruction tuning (OpenAssistant tree, phi family)
//!
//! Falsifier: OpenAssistant tree-format SFT respects assistant-only loss
//! masking — when the longest path is selected, the leaf is an assistant
//! message and forms the loss target.
//!
//! Run with: cargo run --example t3_instruction_openassistant

use apr_cookbook::finetune::instruction_tuning as it;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_tree() -> it::OAMessage {
    it::OAMessage {
        role: "user".into(),
        text: "What is the capital of France?".into(),
        children: vec![
            it::OAMessage {
                role: "assistant".into(),
                text: "Paris.".into(),
                children: vec![],
            },
            it::OAMessage {
                role: "assistant".into(),
                text: "Paris is the capital.".into(),
                children: vec![it::OAMessage {
                    role: "user".into(),
                    text: "And the population?".into(),
                    children: vec![it::OAMessage {
                        role: "assistant".into(),
                        text: "About 2.1 million in the city.".into(),
                        children: vec![],
                    }],
                }],
            },
        ],
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_instruction_openassistant")?;
    let tree = fixture_tree();
    let path = it::oa_flatten_longest_path(&tree);
    println!(
        "✓ OA longest path: {} turns, leaf={:?}",
        path.len(),
        path.last()
    );
    assert!(
        path.len() >= 2,
        "longest path must include ≥1 assistant turn"
    );
    let (last_role, _) = path.last().unwrap();
    assert_eq!(
        last_role, "assistant",
        "leaf must be assistant for loss masking"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let path = it::oa_flatten_longest_path(&fixture_tree());
        assert_eq!(path.last().unwrap().0, "assistant");
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // A tree whose only leaf is a user turn — leaf role is NOT assistant.
        let tree = it::OAMessage {
            role: "user".into(),
            text: "Just a user node.".into(),
            children: vec![],
        };
        let path = it::oa_flatten_longest_path(&tree);
        assert_ne!(path.last().unwrap().0, "assistant");
    }

    #[test]
    fn deterministic_across_runs() {
        let p1 = it::oa_flatten_longest_path(&fixture_tree());
        let p2 = it::oa_flatten_longest_path(&fixture_tree());
        assert_eq!(p1, p2);
    }
}
