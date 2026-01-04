# Inference Explainability

Add explainability to model predictions for transparency and debugging.

## Example

```bash
cargo run --example inference_explainability
```

## Code

```rust
//! Inference Explainability Example
//!
//! Demonstrates adding explainability to model predictions.

use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_explainability")?;

    // Create a model with feature importance tracking
    let model = ExplainableModel::new(ExplainableConfig {
        track_features: true,
        compute_shap: false, // SHAP values (expensive)
        top_k_features: 5,
    });

    // Run inference with explanation
    let input = vec![0.5, 0.3, 0.8, 0.2];
    let (prediction, explanation) = model.predict_with_explanation(&input)?;

    println!("Prediction: {:.4}", prediction);
    println!("\nTop Contributing Features:");
    for (idx, importance) in explanation.top_features(5) {
        println!("  Feature {}: {:.4}", idx, importance);
    }

    ctx.record_float_metric("prediction", prediction as f64);
    ctx.report()?;

    Ok(())
}
```

## Key Concepts

### Feature Importance

Track which input features most influenced the prediction:

```rust
let explanation = model.explain(&input)?;

// Get top 5 most important features
for (feature_idx, importance) in explanation.top_features(5) {
    println!("Feature {}: importance = {:.4}", feature_idx, importance);
}
```

### Attention Visualization

For transformer models, visualize attention patterns:

```rust
let attention = model.attention_weights(&input)?;

// Attention matrix: [num_heads, seq_len, seq_len]
println!("Attention shape: {:?}", attention.shape());
```

## Falsifiable Claims

This recipe supports claim verification for:
- Feature importance computation accuracy
- Attention weight extraction correctness
- Explanation consistency across runs

## Tests

```rust
#[test]
fn test_explainability_deterministic() {
    let model = ExplainableModel::new(Default::default());
    let input = vec![0.5, 0.3, 0.8, 0.2];

    let exp1 = model.explain(&input).unwrap();
    let exp2 = model.explain(&input).unwrap();

    assert_eq!(exp1.top_features(5), exp2.top_features(5));
}
```
