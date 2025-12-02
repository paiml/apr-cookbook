# Bundle a Static Model

Embed a model directly in your Rust binary for single-file deployment.

## Recipe

```rust
use apr_cookbook::bundle::{BundledModel, ModelBundle};

// Step 1: Create the bundle (build script or offline)
fn create_bundle() {
    let weights = std::fs::read("trained_model.bin").unwrap();

    let bundle = ModelBundle::new()
        .with_name("sentiment-classifier")
        .with_description("Binary sentiment classification")
        .with_payload(weights)
        .build();

    std::fs::write("src/models/sentiment.apr", bundle).unwrap();
}

// Step 2: Embed at compile time
const MODEL: &[u8] = include_bytes!("models/sentiment.apr");

// Step 3: Load and use
fn classify(text: &str) -> bool {
    let model = BundledModel::from_bytes(MODEL).unwrap();
    // Use model for inference...
    true
}
```

## Run the Example

```bash
cargo run --example bundle_static_model
```

## Output

```
=== APR Cookbook: Static Model Bundling ===

Creating model bundle...
  Name: sentiment-classifier
  Payload: 1000 bytes

Bundle created: 1048 bytes
  Magic: APRN
  Version: 1.0
  Compressed: no

Loading from bytes...
  Model loaded successfully
  Name: sentiment-classifier
  Size: 1048 bytes

[SUCCESS] Static bundling complete!
```

## Best Practices

1. **Version your models** - Include version in the name
2. **Validate at build time** - Catch errors before deployment
3. **Use compression** - Reduce binary size for large models
