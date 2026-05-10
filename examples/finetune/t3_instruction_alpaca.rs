//! # Tier 3.1 — Instruction tuning (Alpaca format, llama family)
//!
//! Falsifier: Alpaca-format SFT loss-mask. The instruction + input tokens
//! are masked from the loss; only `output` tokens contribute. Closed-form:
//! mask is contiguous `false` then contiguous `true`.
//!
//! Run with: cargo run --example t3_instruction_alpaca

use apr_cookbook::finetune::instruction_tuning as it;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> it::AlpacaRow {
    it::AlpacaRow {
        instruction: "Translate to French".into(),
        input: "Hello world".into(),
        output: "Bonjour le monde".into(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_instruction_alpaca")?;
    let row = fixture();
    let (toks, mask) = it::alpaca_flatten(&row);
    let n_masked = mask.iter().filter(|&&v| !v).count();
    let n_loss = mask.iter().filter(|&&v| v).count();
    println!(
        "✓ Alpaca: {} prefix tokens masked, {} output tokens in loss",
        n_masked, n_loss
    );
    let last_unmasked = mask.iter().rposition(|&v| !v).unwrap();
    let first_masked = mask.iter().position(|&v| v).unwrap();
    assert!(
        last_unmasked < first_masked,
        "mask must transition false→true exactly once"
    );
    assert_eq!(
        n_loss,
        toks.iter()
            .filter(|t| row.output.split_whitespace().any(|o| t == &o))
            .count()
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
        let (_, mask) = it::alpaca_flatten(&fixture());
        let last_unmasked = mask.iter().rposition(|&v| !v).unwrap();
        let first_masked = mask.iter().position(|&v| v).unwrap();
        assert!(last_unmasked < first_masked);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty output → no loss-bearing tokens at all.
        let row = it::AlpacaRow {
            instruction: "x".into(),
            input: "y".into(),
            output: "".into(),
        };
        let (_, mask) = it::alpaca_flatten(&row);
        assert_eq!(mask.iter().filter(|&&v| v).count(), 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let r1 = it::alpaca_flatten(&fixture());
        let r2 = it::alpaca_flatten(&fixture());
        assert_eq!(r1, r2);
    }
}
