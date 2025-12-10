//! # Recipe: Hash Chain Audit Trail
//!
//! **Category**: Inference Monitoring
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: inference-monitoring feature (aprender)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate tamper-evident audit trails using hash chain collectors.
//! Shows how to verify inference history integrity for compliance.
//!
//! ## Toyota Way: 自働化 (Jidoka) - Built-in Quality
//! Tamper-evident chains ensure audit trail integrity automatically.
//!
//! ## Run Command
//! ```bash
//! cargo run --example hash_chain_audit
//! ```

use apr_cookbook::prelude::*;
use aprender::explainable::IntoExplainable;
use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::Estimator;
use entrenar::monitor::inference::{
    path::LinearPath, HashChainCollector, InferenceMonitor, TraceCollector,
};
use serde::Serialize;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("hash_chain_audit")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Demonstrating tamper-evident audit trails with hash chains");
    println!();

    // Train model
    println!("1. Training model...");
    let model = train_model()?;
    let explainable = model.into_explainable();

    // Create hash chain collector for tamper-evident audit
    let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    println!("2. Running inference with hash chain audit...");
    println!();

    // Run several inferences
    let transactions = vec![
        ("TX001", vec![25.0, 1000.0, 1.0]),
        ("TX002", vec![35.0, 5000.0, 2.0]),
        ("TX003", vec![45.0, 10000.0, 3.0]),
        ("TX004", vec![30.0, 2500.0, 1.0]),
        ("TX005", vec![55.0, 15000.0, 5.0]),
    ];

    for (tx_id, features) in &transactions {
        let output = monitor.predict(features, 1);
        println!("  {}: prediction = {:.4}", tx_id, output[0]);
    }
    println!();

    // Verify chain integrity
    println!("3. Verifying hash chain integrity...");
    let verification = monitor.collector().verify_chain();

    println!("   Chain valid: {}", verification.valid);
    println!("   Entries verified: {}", verification.entries_verified);

    if verification.valid {
        println!("   Hash chain integrity: VERIFIED");
    } else {
        println!(
            "   Integrity compromised at index: {:?}",
            verification.first_break
        );
    }
    println!();

    ctx.record_metric("chain_valid", i64::from(verification.valid));
    ctx.record_metric("entries_verified", verification.entries_verified as i64);

    // Export audit trail
    println!("4. Exporting audit trail...");
    let audit_path = ctx.path("hash_chain_audit.json");
    export_hash_chain(&audit_path, monitor.collector())?;
    println!("   Saved to: {:?}", audit_path);

    // Show chain entries
    println!();
    println!("5. Hash chain entries:");
    for entry in monitor.collector().entries().iter().take(3) {
        let prev_hash_hex = hex_encode_prefix(&entry.prev_hash, 8);
        let hash_hex = hex_encode_prefix(&entry.hash, 8);
        println!(
            "   Seq {}: prev_hash={}..., hash={}...",
            entry.sequence, prev_hash_hex, hash_hex
        );
    }
    if monitor.collector().entries().len() > 3 {
        println!("   ... ({} more entries)", monitor.collector().len() - 3);
    }

    Ok(())
}

/// Train a simple linear model for transaction scoring
fn train_model() -> Result<LinearRegression> {
    let x = Matrix::from_vec(
        4,
        3,
        vec![
            20.0, 500.0, 1.0, // Low risk
            40.0, 8000.0, 3.0, // Medium risk
            60.0, 20000.0, 6.0, // High risk
            30.0, 3000.0, 2.0, // Low-medium risk
        ],
    )
    .map_err(|e| CookbookError::Aprender(e.to_string()))?;

    let y = Vector::from_slice(&[0.2, 0.5, 0.9, 0.35]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .map_err(|e| CookbookError::Aprender(e.to_string()))?;

    Ok(model)
}

/// Encode first n bytes of a hash as hex string
fn hex_encode_prefix(bytes: &[u8], n: usize) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(n * 2);
    for b in bytes.iter().take(n) {
        write!(s, "{b:02x}").expect("hex format");
    }
    s
}

/// Encode full hash as hex string
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("hex format");
    }
    s
}

/// Export hash chain to JSON
fn export_hash_chain(
    path: &std::path::Path,
    collector: &HashChainCollector<LinearPath>,
) -> Result<()> {
    #[derive(Serialize)]
    struct ExportEntry {
        sequence: u64,
        timestamp_ns: u64,
        hash: String,
        prev_hash: String,
        output: f32,
    }

    #[derive(Serialize)]
    struct AuditExport {
        chain_length: usize,
        verified: bool,
        entries: Vec<ExportEntry>,
    }

    let verification = collector.verify_chain();

    let entries: Vec<ExportEntry> = collector
        .entries()
        .iter()
        .map(|e| ExportEntry {
            sequence: e.sequence,
            timestamp_ns: e.trace.timestamp_ns,
            hash: hex_encode(&e.hash),
            prev_hash: hex_encode(&e.prev_hash),
            output: e.trace.output,
        })
        .collect();

    let export = AuditExport {
        chain_length: entries.len(),
        verified: verification.valid,
        entries,
    };

    let json = serde_json::to_string_pretty(&export)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_chain_creation() {
        let model = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        let _ = monitor.predict(&[30.0, 2000.0, 2.0], 1);
        let _ = monitor.predict(&[40.0, 5000.0, 3.0], 1);

        assert_eq!(monitor.collector().len(), 2);
    }

    #[test]
    fn test_hash_chain_verification() {
        let model = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        // Add several entries
        for i in 0..5 {
            let _ = monitor.predict(&[(i as f32) * 10.0, 1000.0, 1.0], 1);
        }

        let verification = monitor.collector().verify_chain();

        assert!(verification.valid);
        assert_eq!(verification.entries_verified, 5);
    }

    #[test]
    fn test_hash_chain_linkage() {
        let model = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        let _ = monitor.predict(&[30.0, 2000.0, 2.0], 1);
        let _ = monitor.predict(&[40.0, 5000.0, 3.0], 1);

        let entries = monitor.collector().entries();

        // First entry has all zeros prev_hash
        assert_eq!(entries[0].prev_hash, [0u8; 32]);

        // Second entry links to first
        assert_eq!(entries[1].prev_hash, entries[0].hash);
    }

    #[test]
    fn test_export_hash_chain() {
        let ctx = RecipeContext::new("test_export").unwrap();
        let path = ctx.path("chain.json");

        let model = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        let _ = monitor.predict(&[30.0, 2000.0, 2.0], 1);

        export_hash_chain(&path, monitor.collector()).unwrap();

        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("chain_length"));
        assert!(content.contains("verified"));
        assert!(content.contains("hash"));
    }

    #[test]
    fn test_hash_uniqueness() {
        let model = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        // Different inputs should produce different hashes
        let _ = monitor.predict(&[30.0, 2000.0, 2.0], 1);
        let _ = monitor.predict(&[40.0, 5000.0, 3.0], 1);

        let entries = monitor.collector().entries();
        assert_ne!(entries[0].hash, entries[1].hash);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_chain_always_valid(n in 1usize..20) {
            let model = train_model().unwrap();
            let explainable = model.into_explainable();

            let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
            let mut monitor = InferenceMonitor::new(explainable, collector);

            for i in 0..n {
                let _ = monitor.predict(&[f32::from(i as u8) * 10.0, 1000.0, 1.0], 1);
            }

            let verification = monitor.collector().verify_chain();
            prop_assert!(verification.valid);
            prop_assert_eq!(verification.entries_verified, n);
        }

        #[test]
        fn prop_hash_determinism(
            age in 18.0f32..80.0,
            amount in 100.0f32..50000.0,
            count in 1.0f32..10.0
        ) {
            let model = train_model().unwrap();

            // Run twice with same input
            let explainable1 = model.clone().into_explainable();
            let collector1: HashChainCollector<LinearPath> = HashChainCollector::new();
            let mut monitor1 = InferenceMonitor::new(explainable1, collector1);

            let explainable2 = model.into_explainable();
            let collector2: HashChainCollector<LinearPath> = HashChainCollector::new();
            let mut monitor2 = InferenceMonitor::new(explainable2, collector2);

            let sample = vec![age, amount, count];
            let _ = monitor1.predict(&sample, 1);
            let _ = monitor2.predict(&sample, 1);

            // Outputs should match
            let out1 = monitor1.collector().entries()[0].trace.output;
            let out2 = monitor2.collector().entries()[0].trace.output;

            prop_assert!((out1 - out2).abs() < 1e-6);
        }

        #[test]
        fn prop_sequence_monotonic(n in 2usize..20) {
            let model = train_model().unwrap();
            let explainable = model.into_explainable();

            let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
            let mut monitor = InferenceMonitor::new(explainable, collector);

            for i in 0..n {
                let _ = monitor.predict(&[f32::from(i as u8) * 10.0, 1000.0, 1.0], 1);
            }

            let entries = monitor.collector().entries();
            for i in 1..entries.len() {
                prop_assert!(entries[i].sequence > entries[i-1].sequence);
            }
        }

        #[test]
        fn prop_prev_hash_links_correctly(n in 2usize..10) {
            let model = train_model().unwrap();
            let explainable = model.into_explainable();

            let collector: HashChainCollector<LinearPath> = HashChainCollector::new();
            let mut monitor = InferenceMonitor::new(explainable, collector);

            for i in 0..n {
                let _ = monitor.predict(&[f32::from(i as u8) * 10.0, 1000.0, 1.0], 1);
            }

            let entries = monitor.collector().entries();
            for i in 1..entries.len() {
                prop_assert_eq!(entries[i].prev_hash, entries[i-1].hash);
            }
        }
    }
}
