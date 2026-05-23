//! # Tier 3.1 — Chat template round-trip (qwen3 family)
//!
//! Falsifier: ChatML / Llama2 / Mistral / Phi templates produce token-exact
//! round-trips for ChatML and Phi (lossless), and lossless `[INST]…[/INST]`
//! recovery for Llama2 / Mistral.
//!
//! Run with: cargo run --example t3_instruction_chat_template

use apr_cookbook::finetune::instruction_tuning as it;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn messages() -> Vec<(String, String)> {
    vec![
        ("user".to_string(), "What's 2+2?".to_string()),
        ("assistant".to_string(), "4".to_string()),
        ("user".to_string(), "And 3+3?".to_string()),
    ]
}

fn user_only(messages: &[(String, String)]) -> Vec<(String, String)> {
    messages
        .iter()
        .filter(|(r, _)| r == "user")
        .cloned()
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_instruction_chat_template")?;
    for tpl in [
        it::ChatTemplate::ChatML,
        it::ChatTemplate::Llama2,
        it::ChatTemplate::Mistral,
        it::ChatTemplate::Phi,
    ] {
        let rendered = it::render_chat(tpl, &messages());
        let parsed = it::parse_chat(tpl, &rendered);
        let expected = match tpl {
            it::ChatTemplate::ChatML | it::ChatTemplate::Phi => messages(),
            it::ChatTemplate::Llama2 | it::ChatTemplate::Mistral => user_only(&messages()),
        };
        println!(
            "✓ {:?} round-trip: {} parsed of {} expected",
            tpl,
            parsed.len(),
            expected.len()
        );
        assert_eq!(parsed, expected, "{tpl:?} must round-trip");
    }
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
    fn chatml_falsifier_holds() {
        let m = messages();
        let r = it::render_chat(it::ChatTemplate::ChatML, &m);
        assert_eq!(it::parse_chat(it::ChatTemplate::ChatML, &r), m);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Strip role markers from rendered ChatML output — parser yields nothing.
        let r = it::render_chat(it::ChatTemplate::ChatML, &messages());
        let stripped = r.replace("<|im_start|>", "").replace("<|im_end|>", "");
        assert_ne!(
            it::parse_chat(it::ChatTemplate::ChatML, &stripped),
            messages()
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let r1 = it::render_chat(it::ChatTemplate::ChatML, &messages());
        let r2 = it::render_chat(it::ChatTemplate::ChatML, &messages());
        assert_eq!(r1, r2);
    }
}
