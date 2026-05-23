//! # apr code — Subagent Spawn Payload
//!
//! `apr code` exposes a `Task` tool that spawns subagents. The spawn payload
//! is a JSON envelope with the agent name, prompt, and resource budget. This
//! recipe builds a sample spawn payload via the schema from
//! `apr-code-parity-v1.yaml` row `subagent-spawn`, asserts the shape, and
//! demonstrates the validation logic that `register_task_tool` runs before
//! actually spawning.
//!
//! Demonstrates the **C.4** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-SPAWN-PARITY-001 (SHIPPED v4.4)
//!
//! Run with: cargo run --example code_subagent_spawn_payload
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

/// Validate a Task-tool spawn payload per the `apr code` schema.
/// Returns Ok with the validated subagent name on success.
fn validate_spawn_payload(payload: &Value) -> Result<String> {
    let agent = payload["subagent_type"].as_str().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation(
            "spawn payload missing required `subagent_type` (string)".into(),
        )
    })?;
    let prompt = payload["prompt"].as_str().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation(
            "spawn payload missing required `prompt` (string)".into(),
        )
    })?;
    if prompt.trim().is_empty() {
        return Err(apr_cookbook::CookbookError::Validation(
            "spawn payload `prompt` must not be empty".into(),
        ));
    }
    let _description = payload["description"].as_str().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation(
            "spawn payload missing required `description` (string)".into(),
        )
    })?;
    Ok(agent.to_string())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_subagent_spawn_payload")?;
    let payload = json!({
        "subagent_type": "code-reviewer",
        "description": "review the staged changes",
        "prompt": "Review the staged Rust changes for clippy violations and missing test coverage. Cite each finding with file:line."
    });

    let agent = validate_spawn_payload(&payload)?;
    println!("validated spawn payload for subagent: {agent}");
    println!("(in real `apr code`, this would now POST to the Task tool dispatcher)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_validation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_subagent_type_rejected() {
        let bad = json!({"prompt": "do thing", "description": "x"});
        assert!(validate_spawn_payload(&bad).is_err());
    }

    #[test]
    fn missing_prompt_rejected() {
        let bad = json!({"subagent_type": "x", "description": "x"});
        assert!(validate_spawn_payload(&bad).is_err());
    }

    #[test]
    fn empty_prompt_rejected() {
        let bad = json!({"subagent_type": "x", "description": "x", "prompt": "   "});
        assert!(validate_spawn_payload(&bad).is_err());
    }

    #[test]
    fn missing_description_rejected() {
        let bad = json!({"subagent_type": "x", "prompt": "do thing"});
        assert!(validate_spawn_payload(&bad).is_err());
    }
}
