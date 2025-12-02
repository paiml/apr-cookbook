# Error Handling

Comprehensive error handling with `CookbookError`.

## Error Types

```rust
pub enum CookbookError {
    /// Invalid APR format
    InvalidFormat { message: String },

    /// Model file not found
    ModelNotFound { path: PathBuf },

    /// Feature not available
    FeatureNotAvailable { feature: String },

    /// Dimension mismatch
    DimensionMismatch { expected: Vec<usize>, got: Vec<usize> },

    /// Conversion failed
    ConversionFailed { message: String },

    /// IO error
    Io(std::io::Error),

    /// Aprender error
    Aprender(String),
}
```

## Handling Errors

```rust
use apr_cookbook::{Result, CookbookError};

fn load_model(path: &str) -> Result<Model> {
    let bytes = std::fs::read(path)?;  // Converts io::Error

    let model = BundledModel::from_bytes(&bytes)?;

    if !model.is_compatible() {
        return Err(CookbookError::invalid_format("incompatible version"));
    }

    Ok(model)
}

// Pattern matching
match load_model("model.apr") {
    Ok(model) => println!("Loaded: {}", model.name()),
    Err(CookbookError::ModelNotFound { path }) => {
        eprintln!("File not found: {}", path.display());
    }
    Err(CookbookError::InvalidFormat { message }) => {
        eprintln!("Invalid format: {}", message);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Creating Errors

```rust
// Use helper methods
CookbookError::invalid_format("bad magic bytes")
CookbookError::model_not_found("/path/to/model.apr")
CookbookError::feature_not_available("encryption")
```

## Error Display

All errors implement `Display`:

```rust
let err = CookbookError::invalid_format("bad header");
println!("{}", err);  // "invalid format: bad header"
```
