//! # Recipe: Container Image for Lambda
//!
//! **Category**: Serverless/Lambda
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
//! Package model as container image for Lambda deployment.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serverless_container_image
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("serverless_container_image")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Container image packaging for Lambda");
    println!();

    // Define container layers
    let layers = vec![
        ContainerLayer {
            name: "base".to_string(),
            base_image: "public.ecr.aws/lambda/provided:al2".to_string(),
            size_mb: 50,
        },
        ContainerLayer {
            name: "runtime".to_string(),
            base_image: String::new(),
            size_mb: 20,
        },
        ContainerLayer {
            name: "model".to_string(),
            base_image: String::new(),
            size_mb: 100,
        },
        ContainerLayer {
            name: "application".to_string(),
            base_image: String::new(),
            size_mb: 5,
        },
    ];

    // Build container image
    let mut builder = ContainerBuilder::new("fraud-detector-lambda");

    println!("Building container layers:");
    for layer in &layers {
        builder.add_layer(layer.clone());
        println!("  + {} ({}MB)", layer.name, layer.size_mb);
    }
    println!();

    let image = builder.build()?;

    ctx.record_metric("total_layers", image.layers.len() as i64);
    ctx.record_metric("total_size_mb", i64::from(image.total_size_mb));

    println!("Container Image:");
    println!("  Name: {}", image.name);
    println!("  Tag: {}", image.tag);
    println!("  Total size: {}MB", image.total_size_mb);
    println!("  Layers: {}", image.layers.len());
    println!();

    // Generate Dockerfile
    let dockerfile = generate_dockerfile(&image);
    println!("Generated Dockerfile:");
    println!("{:-<50}", "");
    for line in dockerfile.lines() {
        println!("  {}", line);
    }
    println!("{:-<50}", "");

    // Image optimization analysis
    let analysis = analyze_image(&image);
    println!();
    println!("Optimization Analysis:");
    println!(
        "  Base image overhead: {}MB ({:.1}%)",
        analysis.base_overhead_mb, analysis.base_overhead_pct
    );
    println!(
        "  Model layer: {}MB ({:.1}%)",
        analysis.model_size_mb, analysis.model_pct
    );
    println!(
        "  Cold start impact: {}ms (estimated)",
        analysis.cold_start_impact_ms
    );

    ctx.record_float_metric("model_pct", analysis.model_pct);

    // Save artifacts
    let dockerfile_path = ctx.path("Dockerfile");
    std::fs::write(&dockerfile_path, &dockerfile)?;

    let config_path = ctx.path("container_config.json");
    image.save(&config_path)?;

    println!();
    println!("Dockerfile saved to: {:?}", dockerfile_path);
    println!("Config saved to: {:?}", config_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContainerLayer {
    name: String,
    base_image: String,
    size_mb: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContainerImage {
    name: String,
    tag: String,
    layers: Vec<ContainerLayer>,
    total_size_mb: u32,
}

impl ContainerImage {
    fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[derive(Debug)]
struct ContainerBuilder {
    name: String,
    layers: Vec<ContainerLayer>,
}

impl ContainerBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            layers: Vec::new(),
        }
    }

    fn add_layer(&mut self, layer: ContainerLayer) {
        self.layers.push(layer);
    }

    fn build(self) -> Result<ContainerImage> {
        let total_size: u32 = self.layers.iter().map(|l| l.size_mb).sum();

        Ok(ContainerImage {
            name: self.name,
            tag: "latest".to_string(),
            layers: self.layers,
            total_size_mb: total_size,
        })
    }
}

#[derive(Debug)]
struct ImageAnalysis {
    base_overhead_mb: u32,
    base_overhead_pct: f64,
    model_size_mb: u32,
    model_pct: f64,
    cold_start_impact_ms: u32,
}

fn generate_dockerfile(image: &ContainerImage) -> String {
    let base_layer = image.layers.first();
    let base_image = base_layer
        .map_or("public.ecr.aws/lambda/provided:al2", |l| l.base_image.as_str());

    let mut dockerfile = String::new();
    dockerfile.push_str(&format!("FROM {}\n\n", base_image));
    dockerfile.push_str("# Runtime dependencies\n");
    dockerfile.push_str("COPY bootstrap /var/runtime/\n\n");
    dockerfile.push_str("# Model artifacts\n");
    dockerfile.push_str("COPY model.apr /opt/model/\n\n");
    dockerfile.push_str("# Application binary\n");
    dockerfile.push_str("COPY target/release/handler /var/task/\n\n");
    dockerfile.push_str("# Set entrypoint\n");
    dockerfile.push_str("ENTRYPOINT [\"/var/task/handler\"]\n");

    dockerfile
}

fn analyze_image(image: &ContainerImage) -> ImageAnalysis {
    let base_size = image.layers.first().map_or(0, |l| l.size_mb);
    let model_size = image
        .layers
        .iter()
        .find(|l| l.name == "model")
        .map_or(0, |l| l.size_mb);

    let total = f64::from(image.total_size_mb);

    ImageAnalysis {
        base_overhead_mb: base_size,
        base_overhead_pct: (f64::from(base_size) / total) * 100.0,
        model_size_mb: model_size,
        model_pct: (f64::from(model_size) / total) * 100.0,
        cold_start_impact_ms: image.total_size_mb * 2, // ~2ms per MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_builder() {
        let mut builder = ContainerBuilder::new("test");
        builder.add_layer(ContainerLayer {
            name: "base".to_string(),
            base_image: "alpine".to_string(),
            size_mb: 10,
        });

        let image = builder.build().unwrap();

        assert_eq!(image.name, "test");
        assert_eq!(image.layers.len(), 1);
        assert_eq!(image.total_size_mb, 10);
    }

    #[test]
    fn test_total_size_calculation() {
        let mut builder = ContainerBuilder::new("test");
        builder.add_layer(ContainerLayer {
            name: "a".to_string(),
            base_image: "".to_string(),
            size_mb: 10,
        });
        builder.add_layer(ContainerLayer {
            name: "b".to_string(),
            base_image: "".to_string(),
            size_mb: 20,
        });

        let image = builder.build().unwrap();

        assert_eq!(image.total_size_mb, 30);
    }

    #[test]
    fn test_dockerfile_generation() {
        let image = ContainerImage {
            name: "test".to_string(),
            tag: "latest".to_string(),
            layers: vec![ContainerLayer {
                name: "base".to_string(),
                base_image: "alpine:latest".to_string(),
                size_mb: 5,
            }],
            total_size_mb: 5,
        };

        let dockerfile = generate_dockerfile(&image);

        assert!(dockerfile.contains("FROM alpine:latest"));
        assert!(dockerfile.contains("ENTRYPOINT"));
    }

    #[test]
    fn test_image_analysis() {
        let image = ContainerImage {
            name: "test".to_string(),
            tag: "latest".to_string(),
            layers: vec![
                ContainerLayer {
                    name: "base".to_string(),
                    base_image: "".to_string(),
                    size_mb: 50,
                },
                ContainerLayer {
                    name: "model".to_string(),
                    base_image: "".to_string(),
                    size_mb: 100,
                },
            ],
            total_size_mb: 150,
        };

        let analysis = analyze_image(&image);

        assert_eq!(analysis.base_overhead_mb, 50);
        assert_eq!(analysis.model_size_mb, 100);
        assert!((analysis.model_pct - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_save_image() {
        let ctx = RecipeContext::new("test_container_save").unwrap();
        let path = ctx.path("image.json");

        let image = ContainerImage {
            name: "test".to_string(),
            tag: "v1".to_string(),
            layers: vec![],
            total_size_mb: 0,
        };

        image.save(&path).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_total_size_sums_layers(sizes in proptest::collection::vec(1u32..100, 1..10)) {
            let mut builder = ContainerBuilder::new("test");

            for (i, size) in sizes.iter().enumerate() {
                builder.add_layer(ContainerLayer {
                    name: format!("layer-{}", i),
                    base_image: "".to_string(),
                    size_mb: *size,
                });
            }

            let image = builder.build().unwrap();
            let expected: u32 = sizes.iter().sum();

            prop_assert_eq!(image.total_size_mb, expected);
        }

        #[test]
        fn prop_layer_count_matches(n in 1usize..20) {
            let mut builder = ContainerBuilder::new("test");

            for i in 0..n {
                builder.add_layer(ContainerLayer {
                    name: format!("layer-{}", i),
                    base_image: "".to_string(),
                    size_mb: 10,
                });
            }

            let image = builder.build().unwrap();
            prop_assert_eq!(image.layers.len(), n);
        }
    }
}
