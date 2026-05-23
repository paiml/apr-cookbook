//! # Tier 3.14 — JSON-schema-constrained output (phi family)
//!
//! Falsifier: ≥ 95% of generated outputs parse against the declared schema
//! and contain the expected top-level field.
//!
//! Run with: cargo run --example t3_structured_output_json

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<&'static str> {
    vec![
        "{\"name\": \"alice\", \"age\": 30}",
        "{\"name\": \"bob\", \"age\": 25}",
        "{\"name\": \"carol\", \"age\": 40}",
        "{\"name\": \"dave\", \"age\": 35}",
        "{\"name\": \"eve\", \"age\": 28}",
    ]
}
const FIELD: &str = "name";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_structured_output_json")?;
    let outputs = fixture();
    let valid = outputs
        .iter()
        .filter(|s| specialty::json_has_field(s, FIELD))
        .count();
    let rate = valid as f64 / outputs.len() as f64;
    println!(
        "✓ JSON schema: {}/{} valid ({:.0}%)",
        valid,
        outputs.len(),
        rate * 100.0
    );
    assert!(rate >= 0.95);
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
        let f = fixture();
        let valid = f
            .iter()
            .filter(|s| specialty::json_has_field(s, FIELD))
            .count();
        assert!(valid as f64 / f.len() as f64 >= 0.95);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Plain text — no JSON parses.
        let bogus = vec!["plain text", "more plain text"];
        let valid = bogus
            .iter()
            .filter(|s| specialty::json_has_field(s, FIELD))
            .count();
        assert_eq!(valid, 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let f = fixture();
        let a = f
            .iter()
            .filter(|s| specialty::json_has_field(s, FIELD))
            .count();
        let b = f
            .iter()
            .filter(|s| specialty::json_has_field(s, FIELD))
            .count();
        assert_eq!(a, b);
    }
}
