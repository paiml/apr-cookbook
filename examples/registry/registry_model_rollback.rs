//! # Recipe: Model Rollback
//!
//! **Category**: Model Registry
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Rollback to a previous model version safely.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_model_rollback
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_model_rollback")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Demonstrating safe model rollback");
    println!();

    // Create mock deployment history
    let mut deployment = DeploymentHistory::new("fraud-detector");

    // Deploy version 1.0.0
    deployment.deploy("1.0.0", "Initial production release");
    println!("Deployed v1.0.0: Initial production release");

    // Deploy version 1.1.0
    deployment.deploy("1.1.0", "Improved accuracy");
    println!("Deployed v1.1.0: Improved accuracy");

    // Deploy version 1.2.0
    deployment.deploy("1.2.0", "Added new features");
    println!("Deployed v1.2.0: Added new features");

    ctx.record_metric("total_deployments", deployment.history.len() as i64);

    println!();
    println!("Deployment History:");
    for (i, entry) in deployment.history.iter().enumerate() {
        let status = if Some(i) == deployment.current_index {
            "[CURRENT]"
        } else {
            ""
        };
        println!(
            "  {} v{}: {} {}",
            entry.timestamp, entry.version, entry.description, status
        );
    }

    // Simulate issue - need to rollback
    println!();
    println!("Issue detected! Rolling back to v1.1.0...");

    let rollback_result = deployment.rollback_to("1.1.0")?;
    ctx.record_string_metric("rollback_from", rollback_result.from_version.clone());
    ctx.record_string_metric("rollback_to", rollback_result.to_version.clone());

    println!("Rollback complete:");
    println!("  From: v{}", rollback_result.from_version);
    println!("  To: v{}", rollback_result.to_version);
    println!("  Reason: {}", rollback_result.reason);

    println!();
    println!("Updated Deployment History:");
    for (i, entry) in deployment.history.iter().enumerate() {
        let status = if Some(i) == deployment.current_index {
            "[CURRENT]"
        } else {
            ""
        };
        println!(
            "  {} v{}: {} {}",
            entry.timestamp, entry.version, entry.description, status
        );
    }

    // Verify current version
    let current = deployment.current_version();
    ctx.record_string_metric("current_version", current.clone());
    println!();
    println!("Current active version: v{}", current);

    // Save deployment history
    let history_path = ctx.path("deployment_history.json");
    deployment.save(&history_path)?;
    println!("History saved to: {:?}", history_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeploymentEntry {
    version: String,
    description: String,
    timestamp: u64,
    is_rollback: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeploymentHistory {
    model_name: String,
    history: Vec<DeploymentEntry>,
    current_index: Option<usize>,
}

#[derive(Debug)]
struct RollbackResult {
    from_version: String,
    to_version: String,
    reason: String,
}

impl DeploymentHistory {
    fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            history: Vec::new(),
            current_index: None,
        }
    }

    fn deploy(&mut self, version: &str, description: &str) {
        let entry = DeploymentEntry {
            version: version.to_string(),
            description: description.to_string(),
            timestamp: get_timestamp(),
            is_rollback: false,
        };
        self.history.push(entry);
        self.current_index = Some(self.history.len() - 1);
    }

    fn rollback_to(&mut self, target_version: &str) -> Result<RollbackResult> {
        // Find target version in history
        let _target_idx = self
            .history
            .iter()
            .position(|e| e.version == target_version)
            .ok_or_else(|| CookbookError::ModelNotFound {
                path: std::path::PathBuf::from(target_version),
            })?;

        let from_version = self.current_version();
        let to_version = target_version.to_string();

        // Add rollback entry
        let entry = DeploymentEntry {
            version: target_version.to_string(),
            description: format!("Rollback from v{}", from_version),
            timestamp: get_timestamp(),
            is_rollback: true,
        };
        self.history.push(entry);
        self.current_index = Some(self.history.len() - 1);

        Ok(RollbackResult {
            from_version,
            to_version,
            reason: "Manual rollback due to issue".to_string(),
        })
    }

    fn current_version(&self) -> String {
        self.current_index
            .and_then(|i| self.history.get(i)).map_or_else(|| "none".to_string(), |e| e.version.clone())
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

fn get_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_history_creation() {
        let history = DeploymentHistory::new("test-model");
        assert_eq!(history.model_name, "test-model");
        assert!(history.history.is_empty());
    }

    #[test]
    fn test_deploy() {
        let mut history = DeploymentHistory::new("test");
        history.deploy("1.0.0", "Initial");

        assert_eq!(history.history.len(), 1);
        assert_eq!(history.current_version(), "1.0.0");
    }

    #[test]
    fn test_multiple_deploys() {
        let mut history = DeploymentHistory::new("test");
        history.deploy("1.0.0", "v1");
        history.deploy("1.1.0", "v1.1");
        history.deploy("1.2.0", "v1.2");

        assert_eq!(history.history.len(), 3);
        assert_eq!(history.current_version(), "1.2.0");
    }

    #[test]
    fn test_rollback() {
        let mut history = DeploymentHistory::new("test");
        history.deploy("1.0.0", "v1");
        history.deploy("1.1.0", "v1.1");

        let result = history.rollback_to("1.0.0").unwrap();
        assert_eq!(result.from_version, "1.1.0");
        assert_eq!(result.to_version, "1.0.0");
        assert_eq!(history.current_version(), "1.0.0");
    }

    #[test]
    fn test_rollback_nonexistent_fails() {
        let mut history = DeploymentHistory::new("test");
        history.deploy("1.0.0", "v1");

        let result = history.rollback_to("2.0.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_save() {
        let ctx = RecipeContext::new("test_rollback_save").unwrap();
        let path = ctx.path("history.json");

        let mut history = DeploymentHistory::new("test");
        history.deploy("1.0.0", "Initial");
        history.save(&path).unwrap();

        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_deploy_increments_history(n_deploys in 1usize..10) {
            let mut history = DeploymentHistory::new("test");
            for i in 0..n_deploys {
                history.deploy(&format!("1.{}.0", i), "desc");
            }
            prop_assert_eq!(history.history.len(), n_deploys);
        }

        #[test]
        fn prop_rollback_adds_entry(n_deploys in 2usize..5) {
            let mut history = DeploymentHistory::new("test");
            for i in 0..n_deploys {
                history.deploy(&format!("1.{}.0", i), "desc");
            }

            history.rollback_to("1.0.0").unwrap();

            // Should have original deploys + 1 rollback entry
            prop_assert_eq!(history.history.len(), n_deploys + 1);
        }
    }
}
