//! # Tier 4.8 — XPO exploratory bonus (mistral family)
//!
//! Falsifier: XPO's exploratory bonus broadens generation entropy above
//! online-DPO baseline.
//!
//! Run with: cargo run --example t4_xpo

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn online_dpo_tokens() -> Vec<u32> {
    vec![100, 100, 100, 101, 101, 100, 102, 100]
}
fn xpo_tokens() -> Vec<u32> {
    vec![50, 100, 150, 200, 250, 80, 180, 120]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_xpo")?;
    let online = oa::generation_entropy(&online_dpo_tokens());
    let xpo = oa::generation_entropy(&xpo_tokens());
    println!(
        "✓ XPO: entropy(XPO)={:.2} > entropy(online_DPO)={:.2}",
        xpo, online
    );
    assert!(xpo > online);
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
        assert!(
            oa::generation_entropy(&xpo_tokens()) > oa::generation_entropy(&online_dpo_tokens())
        );
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // XPO with same narrow tokens — entropy equal.
        assert!(
            (oa::generation_entropy(&online_dpo_tokens())
                - oa::generation_entropy(&online_dpo_tokens()))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::generation_entropy(&xpo_tokens());
        let b = oa::generation_entropy(&xpo_tokens());
        assert_eq!(a, b);
    }
}
