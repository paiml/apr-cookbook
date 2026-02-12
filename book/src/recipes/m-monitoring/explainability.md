# Inference Explainability

Add explainability to model predictions using entrenar's inference monitoring
and apr-cookbook's `LinearExplainable` adapter.

## Example

```bash
cargo run --example inference_explainability
```

## Code

```rust
use apr_cookbook::explainable::IntoExplainable;
use apr_cookbook::prelude::*;
use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::Estimator;
use entrenar::monitor::inference::{
    path::LinearPath, InferenceMonitor, RingCollector, TraceCollector,
};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_explainability")?;

    // Train a linear regression model
    let mut model = LinearRegression::new();
    // ... fit model with training data ...

    // Wrap with explainability
    let explainable = model.into_explainable();

    // Create monitored inference with ring buffer collector
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    // Predictions are now traced with feature contributions
    let sample = &[35.0, 80000.0, 4.0];
    let output = monitor.predict(sample, 1);

    // Retrieve explanation
    let traces = monitor.collector().recent(1);
    if let Some(trace) = traces.first() {
        println!("Confidence: {:.1}%", trace.path.confidence() * 100.0);
        for (j, &contrib) in trace.path.feature_contributions().iter().enumerate() {
            println!("  Feature {}: {:+.4}", j, contrib);
        }
    }

    Ok(())
}
```

## Key Concepts

### Feature Contributions

The `LinearExplainable` wrapper decomposes each prediction into per-feature
contributions (coefficient * input), making it clear which features drive
the output.

### Monitored Inference

`InferenceMonitor` from entrenar wraps any `Explainable` model and records
every prediction with its decision path into a collector (ring buffer or
hash chain).

### Audit Trail

Save the collected traces as JSON for compliance and debugging:

```rust
let entries = monitor.collector().recent(monitor.collector().len());
let json = serde_json::to_string_pretty(&entries)?;
std::fs::write("audit.json", json)?;
```

## Tests

The example includes unit tests, integration tests, and property-based tests
verifying:
- Feature contributions sum to logit minus intercept
- Confidence is bounded [0, 1]
- Predictions are deterministic
