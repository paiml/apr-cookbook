# Hash Chain Audit

Cryptographic audit trail for inference using BLAKE3 hash chains.

## Example

```bash
cargo run --example hash_chain_audit
```

## Code

```rust
//! Hash Chain Audit Example
//!
//! Demonstrates tamper-evident logging of model predictions.

use apr_cookbook::prelude::*;
use blake3::Hash;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("hash_chain_audit")?;

    // Create audit chain
    let mut chain = AuditChain::new();

    // Log predictions with cryptographic linking
    let inputs = vec![
        vec![0.5, 0.3, 0.8, 0.2],
        vec![0.1, 0.9, 0.4, 0.6],
        vec![0.7, 0.2, 0.5, 0.8],
    ];

    for (i, input) in inputs.iter().enumerate() {
        let prediction = 0.5 + (i as f32) * 0.1; // Simulated prediction

        let entry = chain.append(AuditEntry {
            timestamp: std::time::SystemTime::now(),
            input_hash: blake3::hash(&input.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()),
            prediction,
            model_version: "v1.0.0".to_string(),
        })?;

        println!("Entry {}: hash = {}", i, entry.hash);
    }

    // Verify chain integrity
    let valid = chain.verify()?;
    println!("\nChain integrity: {}", if valid { "VALID" } else { "INVALID" });
    println!("Chain length: {}", chain.len());

    ctx.record_metric("chain_length", chain.len() as i64);
    ctx.report()?;

    Ok(())
}
```

## Key Concepts

### Hash Chain Structure

Each entry contains:
- Previous entry hash (chain link)
- Input data hash
- Prediction output
- Timestamp
- Model version

```
Entry[0] ─hash─► Entry[1] ─hash─► Entry[2] ─hash─► ...
```

### Tamper Detection

Any modification breaks the chain:

```rust
// Attempt to modify an entry
chain.entries[1].prediction = 0.99;

// Verification fails
assert!(!chain.verify()?);
```

### Export for Compliance

```rust
// Export audit log for regulatory review
let json = chain.to_json()?;
std::fs::write("audit_log.json", json)?;

// Or export with signatures
let signed = chain.sign(&signing_key)?;
std::fs::write("audit_log.signed", signed)?;
```

## Falsifiable Claims

- Chain verification detects any modification
- Hash computation is deterministic
- Append operation is O(1)

## Tests

```rust
#[test]
fn test_chain_tamper_detection() {
    let mut chain = AuditChain::new();

    chain.append(AuditEntry::new(&[0.5], 0.7)).unwrap();
    chain.append(AuditEntry::new(&[0.3], 0.8)).unwrap();

    // Tamper with entry
    chain.entries[0].prediction = 0.99;

    // Should detect tampering
    assert!(!chain.verify().unwrap());
}
```
