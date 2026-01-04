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
use aprender::monitoring::{ExplainabilityReport, AuditChain};
use aprender::apr::AprModel;

// Load model with monitoring enabled
let model = AprModel::load("model.apr")?
    .with_explainability(true)
    .with_audit_chain(true);

// Get explanation with prediction
let (prediction, explanation) = model.predict_with_explanation(&input)?;
println!("Prediction: {:?}", prediction);
println!("Top features: {:?}", explanation.top_features(5));

// Audit chain entry is automatically recorded
let audit = model.audit_chain();
println!("Audit entries: {}", audit.len());
```

## Toyota Way Principles

- **Jidoka**: Built-in quality through explainability
- **Genchi Genbutsu**: "Go and see" via audit trails
- **Kaizen**: Continuous improvement through monitoring
