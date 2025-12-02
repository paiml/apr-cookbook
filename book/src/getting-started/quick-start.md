# Quick Start

Bundle and load your first APR model in 5 minutes.

## Step 1: Create a Model Bundle

```rust
use apr_cookbook::bundle::ModelBundle;

fn main() {
    // Your model weights (from training or file)
    let weights: Vec<u8> = vec![/* your model bytes */];

    // Create APR bundle
    let bundle = ModelBundle::new()
        .with_name("my-classifier")
        .with_description("Sentiment classifier v1.0")
        .with_compression(true)
        .with_payload(weights)
        .build();

    // Save to file
    std::fs::write("model.apr", &bundle).unwrap();
    println!("Saved: {} bytes", bundle.len());
}
```

## Step 2: Load at Runtime

```rust
use apr_cookbook::bundle::BundledModel;

fn main() -> apr_cookbook::Result<()> {
    // Load from file
    let bytes = std::fs::read("model.apr")?;
    let model = BundledModel::from_bytes(&bytes)?;

    println!("Name: {}", model.name());
    println!("Size: {} bytes", model.size());
    println!("Compressed: {}", model.is_compressed());

    Ok(())
}
```

## Step 3: Embed at Compile Time

For production, embed the model directly in your binary:

```rust
use apr_cookbook::bundle::BundledModel;

// Embed at compile time - zero runtime file I/O
const MODEL_BYTES: &[u8] = include_bytes!("../models/classifier.apr");

fn load_model() -> apr_cookbook::Result<BundledModel<'static>> {
    BundledModel::from_bytes(MODEL_BYTES)
}
```

## What's Next?

- [Bundle with Quantization](../recipes/bundle-quantized.md) - Reduce model size
- [Encrypt a Model](../recipes/encrypt-model.md) - Protect proprietary models
- [Convert from SafeTensors](../recipes/convert-safetensors.md) - Import existing models
