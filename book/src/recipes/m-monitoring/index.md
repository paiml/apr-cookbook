# Category M: Inference Monitoring

This category covers monitoring and auditing inference pipelines for production ML systems.

## Recipes

| Recipe | Description |
|--------|-------------|
| [Inference Explainability](./explainability.md) | Add explainability to model predictions |
| [Hash Chain Audit](./hash-chain-audit.md) | Cryptographic audit trail for inference |

## Key Concepts

### Inference Explainability

Understanding why a model made a particular prediction is critical for:
- Debugging model behavior
- Regulatory compliance (GDPR, AI Act)
- Building user trust
- Identifying bias and drift

### Hash Chain Auditing

Cryptographic hash chains provide:
- Tamper-evident logs of all predictions
- Reproducibility verification
- Compliance audit trails
- Data lineage tracking

## Stack Integration

```rust
use apr_cookbook::explainable::IntoExplainable;
use aprender::linear_model::LinearRegression;
use entrenar::monitor::inference::{
    path::LinearPath, InferenceMonitor, RingCollector,
};

// Train and wrap with explainability
let model = LinearRegression::new();
// ... fit model ...
let explainable = model.into_explainable();

// Create monitored inference
let collector: RingCollector<LinearPath, 64> = RingCollector::new();
let mut monitor = InferenceMonitor::new(explainable, collector);

// Predictions are now traced
let output = monitor.predict(&features, 1);
let trace = monitor.collector().recent(1)[0];
println!("{}", trace.path.explain());
```

## Toyota Way Principles

- **Jidoka**: Built-in quality through explainability
- **Genchi Genbutsu**: "Go and see" via audit trails
- **Kaizen**: Continuous improvement through monitoring
