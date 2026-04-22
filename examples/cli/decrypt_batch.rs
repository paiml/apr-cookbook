//! # Recipe: Batch Decrypt with Progress
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr decrypt --batch dir/ --progress --workers 4`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example decrypt_batch` exits 0
//! 2. [x] `cargo test --example decrypt_batch` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] No `unwrap` in main logic
//! 9. [x] Simulates `apr decrypt` batch in-process (no shell-out)
//! 10. [x] Unit tests cover all-success, partial-failure, progress events
//!
//! ## Learning Objective
//! Demonstrates batch decryption of multiple encrypted model files with a
//! deterministic progress reporter. Mixes successful and failed entries (bad
//! key) to show partial-failure semantics -- the CLI must keep going but
//! surface all errors in the final report.
//!
//! ## Run Command
//! ```bash
//! cargo run --example decrypt_batch
//! ```
//!
//! ## References
//! - Percival, C. (2009). *Stronger Key Derivation via Sequential Memory-Hard Functions*. BSDCan.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Pedagogical XOR cipher (blake3 KDF, same shape as decrypt_key_rotation)
// ---------------------------------------------------------------------------

fn derive_stream(key: &[u8], nonce: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len);
    let mut counter: u64 = 0;
    while out.len() < out_len {
        let mut hasher = blake3::Hasher::new();
        hasher.update(key);
        hasher.update(nonce);
        hasher.update(&counter.to_le_bytes());
        let block = hasher.finalize();
        out.extend_from_slice(block.as_bytes());
        counter += 1;
    }
    out.truncate(out_len);
    out
}

fn xor_cipher(data: &[u8], key: &[u8], nonce: &[u8]) -> Vec<u8> {
    let keystream = derive_stream(key, nonce, data.len());
    data.iter()
        .zip(keystream.iter())
        .map(|(d, k)| d ^ k)
        .collect()
}

// Encrypt-then-append-checksum. Decrypt verifies; a mismatch means bad key.
fn encrypt_with_mac(plaintext: &[u8], key: &[u8], nonce: &[u8]) -> Vec<u8> {
    let cipher = xor_cipher(plaintext, key, nonce);
    let mac = blake3::hash(plaintext);
    let mut out = cipher;
    out.extend_from_slice(mac.as_bytes());
    out
}

fn decrypt_with_mac(envelope: &[u8], key: &[u8], nonce: &[u8]) -> Option<Vec<u8>> {
    if envelope.len() < 32 {
        return None;
    }
    let (cipher, mac) = envelope.split_at(envelope.len() - 32);
    let plaintext = xor_cipher(cipher, key, nonce);
    let computed = blake3::hash(&plaintext);
    if computed.as_bytes() == mac {
        Some(plaintext)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecryptJob {
    name: String,
    envelope: Vec<u8>,
    key: Vec<u8>,
    nonce: Vec<u8>,
}

#[derive(Debug, Clone)]
struct DecryptOutcome {
    name: String,
    ok: bool,
    plaintext_bytes: usize,
    error: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct BatchSummary {
    total: usize,
    ok: usize,
    failed: usize,
    progress_events: usize,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn run_batch<F>(jobs: &[DecryptJob], mut progress: F) -> (Vec<DecryptOutcome>, BatchSummary)
where
    F: FnMut(usize, usize, &str),
{
    let mut outcomes = Vec::with_capacity(jobs.len());
    let mut summary = BatchSummary {
        total: jobs.len(),
        ..Default::default()
    };
    for (i, job) in jobs.iter().enumerate() {
        progress(i + 1, jobs.len(), &job.name);
        summary.progress_events += 1;

        if let Some(pt) = decrypt_with_mac(&job.envelope, &job.key, &job.nonce) {
            summary.ok += 1;
            outcomes.push(DecryptOutcome {
                name: job.name.clone(),
                ok: true,
                plaintext_bytes: pt.len(),
                error: None,
            });
        } else {
            summary.failed += 1;
            outcomes.push(DecryptOutcome {
                name: job.name.clone(),
                ok: false,
                plaintext_bytes: 0,
                error: Some("MAC verification failed".into()),
            });
        }
    }
    (outcomes, summary)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("decrypt_batch")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seed = hash_name_to_seed("decrypt-batch");
    let correct_key = b"correct-master-key-0123";
    let wrong_key = b"wrong-master-key-9999";

    // Build four encrypted envelopes with the correct key.
    let mut jobs = Vec::new();
    let names = ["alpha", "beta", "gamma", "delta"];
    for name in &names {
        let plaintext = generate_model_payload(seed ^ hash_name_to_seed(name), 64);
        let nonce_str = format!("nonce-{name}");
        let envelope = encrypt_with_mac(&plaintext, correct_key, nonce_str.as_bytes());
        let out_path = ctx.path(&format!("{name}.enc"));
        std::fs::write(&out_path, &envelope)?;
        // Three of four use the correct key; one gets the wrong key -> MAC fails.
        let use_wrong = *name == "gamma";
        jobs.push(DecryptJob {
            name: (*name).to_string(),
            envelope,
            key: if use_wrong {
                wrong_key.to_vec()
            } else {
                correct_key.to_vec()
            },
            nonce: nonce_str.as_bytes().to_vec(),
        });
    }

    println!("Dispatching {} decrypt jobs", jobs.len());

    let (outcomes, summary) = run_batch(&jobs, |i, n, name| {
        println!("  [{i}/{n}] decrypting {name}");
    });

    println!("\n--- Outcomes ---");
    println!("{:>10} {:>6} {:>14} Error", "Name", "OK", "PlaintextLen");
    for o in &outcomes {
        println!(
            "{:>10} {:>6} {:>14} {}",
            o.name,
            o.ok,
            o.plaintext_bytes,
            o.error.as_deref().unwrap_or("")
        );
    }
    println!(
        "\nSummary: total={}, ok={}, failed={}, progress_events={}",
        summary.total, summary.ok, summary.failed, summary.progress_events
    );

    // Sanity.
    assert_eq!(summary.total, 4);
    assert_eq!(summary.ok, 3);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.progress_events, 4);

    let out = json!({
        "recipe": ctx.name(),
        "summary": {
            "total": summary.total,
            "ok": summary.ok,
            "failed": summary.failed,
            "progress_events": summary.progress_events,
        },
        "outcomes": outcomes.iter().map(|o| json!({
            "name": o.name,
            "ok": o.ok,
            "plaintext_bytes": o.plaintext_bytes,
            "error": o.error,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("batch-decrypt.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(name: &str, plaintext: &[u8], ek: &[u8], dk: &[u8]) -> DecryptJob {
        let envelope = encrypt_with_mac(plaintext, ek, b"n");
        DecryptJob {
            name: name.into(),
            envelope,
            key: dk.to_vec(),
            nonce: b"n".to_vec(),
        }
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let pt = b"hello";
        let env = encrypt_with_mac(pt, b"k", b"n");
        let back = decrypt_with_mac(&env, b"k", b"n").expect("ok");
        assert_eq!(back, pt);
    }

    #[test]
    fn test_wrong_key_fails_mac() {
        let env = encrypt_with_mac(b"hello", b"k1", b"n");
        assert!(decrypt_with_mac(&env, b"k2", b"n").is_none());
    }

    #[test]
    fn test_short_envelope_returns_none() {
        assert!(decrypt_with_mac(&[0_u8; 10], b"k", b"n").is_none());
    }

    #[test]
    fn test_run_batch_all_ok() {
        let jobs = vec![
            make_job("a", b"aa", b"k", b"k"),
            make_job("b", b"bb", b"k", b"k"),
        ];
        let mut events = 0;
        let (outcomes, s) = run_batch(&jobs, |_, _, _| events += 1);
        assert!(outcomes.iter().all(|o| o.ok));
        assert_eq!(s.ok, 2);
        assert_eq!(s.failed, 0);
        assert_eq!(events, 2);
    }

    #[test]
    fn test_run_batch_partial_failure() {
        let jobs = vec![
            make_job("good", b"pt1", b"k", b"k"),
            make_job("bad", b"pt2", b"k", b"wrong"),
        ];
        let (outcomes, s) = run_batch(&jobs, |_, _, _| {});
        assert_eq!(s.total, 2);
        assert_eq!(s.ok, 1);
        assert_eq!(s.failed, 1);
        assert!(outcomes.iter().any(|o| !o.ok));
    }

    #[test]
    fn test_empty_batch() {
        let (outcomes, s) = run_batch(&[], |_, _, _| {});
        assert!(outcomes.is_empty());
        assert_eq!(s.total, 0);
    }
}
