//! In-process contract validation for every YAML under `contracts/`.
//!
//! Replaces the external `pv validate` / `pv lint` CLI dependency with a direct
//! Rust call through the `aprender-contracts` crate (published as lib
//! `provable_contracts` in the aprender monorepo, v0.31.2+).
//!
//! This test is load-bearing for two coverage claims in
//! `docs/specifications/components/quality-gates.md`:
//!
//! 1. Every contract YAML must parse against the canonical schema.
//! 2. Every contract YAML must pass `validate_contract` (checks obligation wiring,
//!    kernel bindings, schema completeness).
//!
//! Run: `cargo test --test contracts`

use std::path::Path;

use provable_contracts::schema::{parse_contract, validate_contract};

/// Every YAML under `contracts/` that the cookbook owns.
///
/// Kept as a static list rather than a directory walk so that adding a new contract
/// is a deliberate edit to this file — new contracts must come with a test entry,
/// which forces reviewers to see them.
const CONTRACT_FILES: &[&str] = &[
    "aes256-gcm-decrypt-v1.yaml",
    "apr-format-roundtrip-v1.yaml",
    "avx512-matmul-v1.yaml",
    "cli-parity-v1.yaml",
    "docs-schema-v1.yaml",
    "flash-attention-v1.yaml",
    // architecture-demos (PMAT-300+): one per family + cross-family detector (PMAT-309).
    "inference-arch-detector-v1.yaml",
    "inference-arch-summary-v1.yaml",
    "inference-bert-smoke-v1.yaml",
    "inference-deepseek-smoke-v1.yaml",
    "inference-falcon-h1-smoke-v1.yaml",
    "inference-gemma-smoke-v1.yaml",
    "inference-gpt2-smoke-v1.yaml",
    "inference-gptneox-smoke-v1.yaml",
    "inference-llama-smoke-v1.yaml",
    "inference-mamba-smoke-v1.yaml",
    "inference-mistral-smoke-v1.yaml",
    "inference-moonshine-smoke-v1.yaml",
    "inference-openelm-smoke-v1.yaml",
    "inference-opt-smoke-v1.yaml",
    "inference-phi-smoke-v1.yaml",
    "inference-qwen2-smoke-v1.yaml",
    "inference-qwen3-5-smoke-v1.yaml",
    "inference-qwen3-smoke-v1.yaml",
    "inference-rwkv7-smoke-v1.yaml",
    "int4-quantization-v1.yaml",
    "lz4-decompression-v1.yaml",
    "mmap-inference-v1.yaml",
    "recipe-iiur-config-v1.yaml",
    "recipe-iiur-v1.yaml",
    "whisper-wer-v1.yaml",
];

fn contract_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("contracts")
        .join(name)
}

#[test]
fn every_contract_parses() {
    let mut failures = Vec::new();
    for &name in CONTRACT_FILES {
        let path = contract_path(name);
        match parse_contract(&path) {
            Ok(_) => println!("  OK: {name}"),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "contract YAML parse failures:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn every_contract_validates() {
    let mut failures = Vec::new();
    for &name in CONTRACT_FILES {
        let path = contract_path(name);
        let contract = match parse_contract(&path) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{name}: parse failed: {e}"));
                continue;
            }
        };
        let violations = validate_contract(&contract);
        if violations.is_empty() {
            println!("  VALID: {name}");
        } else {
            failures.push(format!(
                "{name}: {} violations: {}",
                violations.len(),
                violations
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "contract validation failures:\n  {}",
        failures.join("\n  ")
    );
}

/// Every contract on disk must appear in `CONTRACT_FILES`. Guards against
/// silently-added YAMLs that bypass the reviewer's attention.
#[test]
fn contract_inventory_matches_disk() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("read contracts/")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            // YAML extension is case-sensitive in this project — reject `.YAML`, `.Yaml`.
            Path::new(n).extension().is_some_and(|e| e == "yaml")
        })
        .filter(|n| {
            // `binding.yaml` is the pv binding registry (contract → Rust kernel map),
            // not a contract definition. It lives alongside contracts by `pv` convention
            // but isn't in CONTRACT_FILES.
            n != "binding.yaml"
        })
        .collect();
    on_disk.sort();

    let mut expected: Vec<String> = CONTRACT_FILES.iter().map(|s| (*s).into()).collect();
    expected.sort();

    assert_eq!(
        on_disk, expected,
        "contract inventory drift:\n  on disk: {on_disk:?}\n  expected: {expected:?}\n\
         If a new contract was added, update CONTRACT_FILES in tests/contracts.rs."
    );
}
