//! # Tier 3.1 — Instruction tuning (ShareGPT multi-turn, mistral family)
//!
//! Falsifier: ShareGPT multi-turn SFT preserves multi-turn coherence —
//! render → parse must round-trip exactly to the original turn list.
//!
//! Run with: cargo run --example t3_instruction_sharegpt

use apr_cookbook::finetune::instruction_tuning as it;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_turns() -> Vec<it::ShareGPTTurn> {
    vec![
        it::ShareGPTTurn {
            role: "user".into(),
            content: "what is 2+2?".into(),
        },
        it::ShareGPTTurn {
            role: "assistant".into(),
            content: "4".into(),
        },
        it::ShareGPTTurn {
            role: "user".into(),
            content: "and 3+3?".into(),
        },
        it::ShareGPTTurn {
            role: "assistant".into(),
            content: "6".into(),
        },
        it::ShareGPTTurn {
            role: "user".into(),
            content: "9+9?".into(),
        },
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_instruction_sharegpt")?;
    let turns = fixture_turns();
    let flat = it::sharegpt_render(&turns);
    let parsed = it::sharegpt_parse(&flat);
    println!(
        "✓ ShareGPT 5-turn round-trip: {} turns rendered, {} parsed",
        turns.len(),
        parsed.len()
    );
    assert_eq!(parsed, turns, "5-turn ShareGPT must round-trip");
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
        let turns = fixture_turns();
        assert_eq!(it::sharegpt_parse(&it::sharegpt_render(&turns)), turns);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Lose role markers → parser drops the content (different from input).
        let bogus = "what is 2+2? 4 and 3+3? 6";
        let parsed = it::sharegpt_parse(bogus);
        assert!(parsed.is_empty());
    }

    #[test]
    fn deterministic_across_runs() {
        let t = fixture_turns();
        let r1 = it::sharegpt_parse(&it::sharegpt_render(&t));
        let r2 = it::sharegpt_parse(&it::sharegpt_render(&t));
        assert_eq!(r1, r2);
    }
}
