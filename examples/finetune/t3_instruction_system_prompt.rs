//! # Tier 3.1 — System-prompt SFT (gemma family)
//!
//! Falsifier: system-prompt prefix is masked from loss but visible to the
//! forward pass. Closed-form: system-prompt tokens have mask=false; body
//! tokens have mask=true; both contribute to the input sequence.
//!
//! Run with: cargo run --example t3_instruction_system_prompt

use apr_cookbook::finetune::instruction_tuning as it;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SYSTEM: &str = "You are a careful tutor who answers in one sentence.";
const BODY: &str = "Q: What is 2+2? A: It is 4.";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_instruction_system_prompt")?;
    let (toks, mask) = it::system_prompt_mask(SYSTEM, BODY);
    let n_system = SYSTEM.split_whitespace().count();
    let n_body = BODY.split_whitespace().count();
    println!(
        "✓ system-prompt mask: {} system tokens (loss-masked), {} body tokens (loss-bearing)",
        n_system, n_body
    );
    assert_eq!(
        toks.len(),
        n_system + n_body,
        "all tokens visible to forward pass"
    );
    assert_eq!(mask.iter().filter(|&&v| !v).count(), n_system);
    assert_eq!(mask.iter().filter(|&&v| v).count(), n_body);
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
        let (_, mask) = it::system_prompt_mask(SYSTEM, BODY);
        assert!(mask.iter().filter(|&&v| !v).count() > 0);
        assert!(mask.iter().filter(|&&v| v).count() > 0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty system → no mask=false entries.
        let (_, mask) = it::system_prompt_mask("", BODY);
        assert_eq!(mask.iter().filter(|&&v| !v).count(), 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let r1 = it::system_prompt_mask(SYSTEM, BODY);
        let r2 = it::system_prompt_mask(SYSTEM, BODY);
        assert_eq!(r1, r2);
    }
}
