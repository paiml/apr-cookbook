# Custom Model Types

Define custom model types for specialized use cases.

## Implementing a Custom Model

```rust
use serde::{Serialize, Deserialize};
use aprender::format::{save, load, ModelType, SaveOptions};

#[derive(Debug, Serialize, Deserialize)]
struct MyCustomModel {
    layers: Vec<Layer>,
    vocab_size: usize,
    hidden_dim: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl MyCustomModel {
    fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            layers: vec![],
            vocab_size,
            hidden_dim,
        }
    }

    fn save(&self, path: &str) -> apr_cookbook::Result<()> {
        save(
            self,
            ModelType::Custom,
            path,
            SaveOptions::default()
                .with_name("my-model")
                .with_compression(true),
        )?;
        Ok(())
    }

    fn load(path: &str) -> apr_cookbook::Result<Self> {
        Ok(load(path, ModelType::Custom)?)
    }
}
```

## Model Type Guidelines

| Model Type | Use Case |
|------------|----------|
| `ModelType::Custom` | Generic serializable structs |
| `ModelType::LinearRegression` | Linear models |
| `ModelType::NeuralNetwork` | Neural network weights |
| `ModelType::Transformer` | Transformer architectures |

## Validation

Always validate loaded models:

```rust
impl MyCustomModel {
    fn validate(&self) -> apr_cookbook::Result<()> {
        if self.layers.is_empty() {
            return Err(apr_cookbook::CookbookError::invalid_format(
                "model has no layers"
            ));
        }
        Ok(())
    }
}
```
