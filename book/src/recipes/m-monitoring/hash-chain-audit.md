# Hash Chain Audit

Tamper-evident audit trail for inference using entrenar's `HashChainCollector`.

## Example

```bash
cargo run --example hash_chain_audit
```

## Code

```rust
use apr_cookbook::explainable::IntoExplainable;
use apr_cookbook::prelude::*;
use aprender::linear_model::LinearRegression;
use entrenar::monitor::inference::{
    path::LinearPath, HashChainCollector, InferenceMonitor, TraceCollector,
};

fn main() -> Result<()> {
    // Train model and wrap with explainability
    let model = train_model()?;
    let explainable = model.into_explainable();

    // Create hash chain collector for tamper-evident audit
    let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    // Run inferences - each one is cryptographically chained
    let _ = monitor.predict(&[25.0, 1000.0, 1.0], 1);
    let _ = monitor.predict(&[35.0, 5000.0, 2.0], 1);

    // Verify chain integrity
    let verification = monitor.collector().verify_chain();
    println!("Chain valid: {}", verification.valid);
    println!("Entries verified: {}", verification.entries_verified);

    Ok(())
}
```

## Key Concepts

### Hash Chain Structure

Each entry contains a BLAKE3 hash linking it to the previous entry:

```
Entry[0] --hash--> Entry[1] --hash--> Entry[2] --hash--> ...
```

### Tamper Detection

Any modification to a historical entry breaks the chain. Verification
traverses the chain and recomputes hashes to detect breaks.

### Export for Compliance

```rust
// Export audit log for regulatory review
let json = serde_json::to_string_pretty(&collector.entries())?;
std::fs::write("audit_log.json", json)?;
```

## Tests

The example includes property-based tests verifying:
- Chain is always valid after sequential appends
- Hash determinism for identical inputs
- Sequence numbers are monotonically increasing
- `prev_hash` correctly links to previous entry's hash
