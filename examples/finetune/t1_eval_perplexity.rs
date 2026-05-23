//! # Tier 1.2 — Eval primitive — Perplexity
//!
//! Compute perplexity on a uniform-vocab probe sequence and assert it
//! equals the vocab size. Falsifier: PPL on uniform p = 1/V is exactly V.
//!
//! Demonstrates the **t1_eval_perplexity** recipe per
//! `docs/specifications/fine-tuning-cookbook.md` (PMAT-332).
//!
//! Run with: cargo run --example t1_eval_perplexity

use apr_cookbook::finetune::eval_primitives;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_eval_perplexity/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_eval_perplexity")?;
    // Probe: each line in data.jsonl is a vocab size; assert PPL = vocab_size.
    let body = std::fs::read_to_string(FIXTURE)
        .map_err(|e| apr_cookbook::CookbookError::invalid_format(format!("read fixture: {e}")))?;
    let mut probes = 0;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vocab_size = parse_vocab(line);
        let ppl = eval_primitives::perplexity_uniform(vocab_size);
        let expected = f64::from(vocab_size);
        let rel_err = (ppl - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "vocab={vocab_size}: PPL={ppl} != {expected} (rel_err={rel_err})"
        );
        probes += 1;
    }
    println!("✓ perplexity uniform-vocab equality: {probes} probes verified");
    Ok(())
}

fn parse_vocab(line: &str) -> u32 {
    line.find("\"vocab_size\":")
        .and_then(|p| {
            let rest = &line[p + 13..];
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            rest[..end].trim().parse().ok()
        })
        .unwrap_or(0)
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
        // PPL on uniform p = 1/V is exactly V (mathematical identity).
        for v in [10u32, 100, 1000, 10000, 50000] {
            let ppl = eval_primitives::perplexity_uniform(v);
            assert!(
                (ppl - f64::from(v)).abs() / f64::from(v) < 1e-10,
                "vocab={v}: PPL={ppl}"
            );
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // If we shift log-likelihoods, PPL changes — confirms the
        // falsifier is non-trivial.
        let log_p = (-(1000.0_f64).ln()) - 1.0; // shift away from uniform
        let lls = vec![log_p; 100];
        let ppl = eval_primitives::perplexity(&lls);
        assert!(
            (ppl - 1000.0).abs() > 100.0,
            "perturbed log-p should yield PPL != vocab_size, got {ppl}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = eval_primitives::perplexity_uniform(1000);
        let b = eval_primitives::perplexity_uniform(1000);
        assert_eq!(a, b);
    }
}
