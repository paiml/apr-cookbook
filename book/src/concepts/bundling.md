# Model Bundling

Bundling converts model weights into the APR format for deployment.

## The ModelBundle Builder

```rust
use apr_cookbook::bundle::ModelBundle;

let bundle = ModelBundle::new()
    .with_name("sentiment-v1")
    .with_description("BERT-based sentiment classifier")
    .with_compression(true)
    .with_payload(model_weights)
    .build();
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `with_name(s)` | Set model name (max 255 chars) |
| `with_description(s)` | Set description (optional) |
| `with_compression(bool)` | Enable zstd compression |
| `with_payload(bytes)` | Set model weights |
| `build()` | Create the APR bundle |

## Loading Bundles

```rust
use apr_cookbook::bundle::BundledModel;

// From bytes (zero-copy)
let model = BundledModel::from_bytes(&bundle_bytes)?;

// Access metadata
println!("Name: {}", model.name());
println!("Version: {:?}", model.version());
println!("Size: {} bytes", model.size());

// Check flags
if model.is_compressed() {
    println!("Payload is compressed");
}
```

## BundledModel Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `name()` | `&str` | Model name |
| `version()` | `(u8, u8)` | Format version |
| `size()` | `usize` | Total size in bytes |
| `is_compressed()` | `bool` | Compression flag |
| `is_encrypted()` | `bool` | Encryption flag |
| `is_signed()` | `bool` | Signature flag |
| `as_bytes()` | `&[u8]` | Raw bundle bytes |

## Compile-Time Embedding

The recommended pattern for production:

```rust
// Embed at compile time
const MODEL: &[u8] = include_bytes!("models/classifier.apr");

fn get_model() -> BundledModel<'static> {
    // This never fails if the file is valid APR
    BundledModel::from_bytes(MODEL).expect("embedded model is valid")
}
```

Benefits:
- No file I/O at runtime
- Model integrity verified at compile time
- Single binary deployment
