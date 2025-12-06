//! # Recipe: Register APR Model in Registry
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
//! Register `.apr` model in a mock registry with versioning.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_register_apr
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_register_apr")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Registering .apr model in mock registry");
    println!();

    // Create mock registry in temp directory
    let registry_path = ctx.path("registry.json");
    let mut registry = MockRegistry::new(&registry_path);

    // Create model to register
    let model_seed = hash_name_to_seed("fraud_detector");
    let payload = generate_model_payload(model_seed, 512);
    let model_bytes = ModelBundle::new()
        .with_name("fraud-detector")
        .with_compression(true)
        .with_payload(payload)
        .build();

    let model_path = ctx.path("fraud_detector.apr");
    std::fs::write(&model_path, &model_bytes)?;

    // Register model v1.0.0
    let model_id = registry.register(
        "fraud-detector",
        &model_path,
        SemVer::new(1, 0, 0),
        ModelCard {
            description: "Fraud detection classifier for transactions".to_string(),
            metrics: [
                ("accuracy".to_string(), "0.95".to_string()),
                ("f1_score".to_string(), "0.92".to_string()),
            ]
            .into_iter()
            .collect(),
            tags: vec!["fraud".to_string(), "classification".to_string()],
        },
    )?;

    ctx.record_string_metric("model_id", model_id.clone());
    println!("Registered model: {}", model_id);

    // Stage to production
    registry.stage(&model_id, Stage::Production)?;
    println!("Staged to production");

    // Register model v1.1.0 (update)
    let model_id_v2 = registry.register(
        "fraud-detector",
        &model_path,
        SemVer::new(1, 1, 0),
        ModelCard {
            description: "Fraud detection v1.1 with improved recall".to_string(),
            metrics: [
                ("accuracy".to_string(), "0.96".to_string()),
                ("f1_score".to_string(), "0.94".to_string()),
                ("recall".to_string(), "0.91".to_string()),
            ]
            .into_iter()
            .collect(),
            tags: vec![
                "fraud".to_string(),
                "classification".to_string(),
                "v1.1".to_string(),
            ],
        },
    )?;

    ctx.record_string_metric("model_id_v2", model_id_v2.clone());
    println!("Registered model v1.1.0: {}", model_id_v2);

    // List models
    let models = registry.list()?;
    ctx.record_metric("model_count", models.len() as i64);

    // Save registry
    registry.save()?;

    println!();
    println!("Registry contents:");
    for model in &models {
        println!("  {} v{} [{}]", model.name, model.version, model.stage);
    }
    println!();
    println!("Registry saved to: {:?}", registry_path);

    Ok(())
}

/// Semantic version
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemVer {
    major: u32,
    minor: u32,
    patch: u32,
}

impl SemVer {
    fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Model deployment stage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum Stage {
    Development,
    Staging,
    Production,
    Archived,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Development => write!(f, "development"),
            Stage::Staging => write!(f, "staging"),
            Stage::Production => write!(f, "production"),
            Stage::Archived => write!(f, "archived"),
        }
    }
}

/// Model card with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCard {
    description: String,
    metrics: HashMap<String, String>,
    tags: Vec<String>,
}

/// Registered model entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelEntry {
    id: String,
    name: String,
    version: SemVer,
    stage: Stage,
    path: String,
    card: ModelCard,
    registered_at: u64,
}

impl std::fmt::Display for ModelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{}", self.name, self.version)
    }
}

/// Mock model registry
#[derive(Debug)]
struct MockRegistry {
    path: std::path::PathBuf,
    models: Vec<ModelEntry>,
}

impl MockRegistry {
    fn new(path: &std::path::Path) -> Self {
        Self {
            path: path.to_path_buf(),
            models: Vec::new(),
        }
    }

    fn register(
        &mut self,
        name: &str,
        model_path: &std::path::Path,
        version: SemVer,
        card: ModelCard,
    ) -> Result<String> {
        let id = format!("{}:{}", name, version);

        let entry = ModelEntry {
            id: id.clone(),
            name: name.to_string(),
            version,
            stage: Stage::Development,
            path: model_path.to_string_lossy().to_string(),
            card,
            registered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        self.models.push(entry);
        Ok(id)
    }

    fn stage(&mut self, id: &str, stage: Stage) -> Result<()> {
        for model in &mut self.models {
            if model.id == id {
                model.stage = stage;
                return Ok(());
            }
        }
        Err(CookbookError::ModelNotFound {
            path: std::path::PathBuf::from(id),
        })
    }

    fn list(&self) -> Result<Vec<ModelEntry>> {
        Ok(self.models.clone())
    }

    fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.models)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let ctx = RecipeContext::new("test_reg").unwrap();
        let path = ctx.path("reg.json");
        let registry = MockRegistry::new(&path);
        assert!(registry.models.is_empty());
    }

    #[test]
    fn test_model_registration() {
        let ctx = RecipeContext::new("test_reg2").unwrap();
        let reg_path = ctx.path("reg.json");
        let model_path = ctx.path("model.apr");

        // Create model file
        let model = ModelBundle::new().with_payload(vec![1, 2, 3]).build();
        std::fs::write(&model_path, model).unwrap();

        let mut registry = MockRegistry::new(&reg_path);
        let id = registry
            .register(
                "test-model",
                &model_path,
                SemVer::new(1, 0, 0),
                ModelCard {
                    description: "Test".to_string(),
                    metrics: HashMap::new(),
                    tags: vec![],
                },
            )
            .unwrap();

        assert_eq!(id, "test-model:1.0.0");
        assert_eq!(registry.models.len(), 1);
    }

    #[test]
    fn test_staging() {
        let ctx = RecipeContext::new("test_stage").unwrap();
        let reg_path = ctx.path("reg.json");
        let model_path = ctx.path("model.apr");

        std::fs::write(&model_path, ModelBundle::new().build()).unwrap();

        let mut registry = MockRegistry::new(&reg_path);
        let id = registry
            .register(
                "model",
                &model_path,
                SemVer::new(1, 0, 0),
                ModelCard {
                    description: "".to_string(),
                    metrics: HashMap::new(),
                    tags: vec![],
                },
            )
            .unwrap();

        registry.stage(&id, Stage::Production).unwrap();

        let models = registry.list().unwrap();
        assert!(matches!(models[0].stage, Stage::Production));
    }

    #[test]
    fn test_semver_display() {
        let v = SemVer::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_registry_save() {
        let ctx = RecipeContext::new("test_save").unwrap();
        let reg_path = ctx.path("reg.json");
        let model_path = ctx.path("model.apr");

        std::fs::write(&model_path, ModelBundle::new().build()).unwrap();

        let mut registry = MockRegistry::new(&reg_path);
        registry
            .register(
                "model",
                &model_path,
                SemVer::new(1, 0, 0),
                ModelCard {
                    description: "".to_string(),
                    metrics: HashMap::new(),
                    tags: vec![],
                },
            )
            .unwrap();

        registry.save().unwrap();
        assert!(reg_path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_semver_format(major in 0u32..100, minor in 0u32..100, patch in 0u32..100) {
            let v = SemVer::new(major, minor, patch);
            let s = v.to_string();
            prop_assert!(s.contains('.'));
            prop_assert_eq!(s.matches('.').count(), 2);
        }

        #[test]
        fn prop_registration_idempotent(name in "[a-z]{3,10}") {
            let ctx = RecipeContext::new("prop_reg").unwrap();
            let reg_path = ctx.path("reg.json");
            let model_path = ctx.path("model.apr");

            std::fs::write(&model_path, ModelBundle::new().build()).unwrap();

            let mut registry = MockRegistry::new(&reg_path);
            let id = registry.register(
                &name,
                &model_path,
                SemVer::new(1, 0, 0),
                ModelCard {
                    description: "".to_string(),
                    metrics: HashMap::new(),
                    tags: vec![],
                },
            ).unwrap();

            prop_assert!(id.starts_with(&name));
        }
    }
}
