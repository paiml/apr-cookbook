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
    "inference-arch-alias-resolver-v1.yaml",
    "inference-arch-compare-v1.yaml",
    "inference-arch-detector-v1.yaml",
    "inference-arch-quirk-audit-v1.yaml",
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
    // architecture-demos v1.2 (PMAT-320): forward-bridge resolution pipeline.
    "inference-arch-resolution-pipeline-v1.yaml",
    // fine-tuning-cookbook Tier 1.1 (PMAT-331): SFT minimal × 5 families.
    "finetune-t1-sft-minimal-llama-v1.yaml",
    "finetune-t1-sft-minimal-mistral-v1.yaml",
    "finetune-t1-sft-minimal-phi-v1.yaml",
    "finetune-t1-sft-minimal-qwen-v1.yaml",
    "finetune-t1-sft-minimal-gemma-v1.yaml",
    // fine-tuning-cookbook Tier 1.2 (PMAT-332): Eval primitives × 5.
    "finetune-t1-eval-perplexity-v1.yaml",
    "finetune-t1-eval-accuracy-v1.yaml",
    "finetune-t1-eval-f1-v1.yaml",
    "finetune-t1-eval-rouge-l-v1.yaml",
    "finetune-t1-eval-bleu-v1.yaml",
    // fine-tuning-cookbook Tier 1.3 (PMAT-333): Tabular regression × 5.
    "finetune-t1-tabular-regression-housing-v1.yaml",
    "finetune-t1-tabular-regression-energy-v1.yaml",
    "finetune-t1-tabular-regression-timeseries-v1.yaml",
    "finetune-t1-tabular-regression-multitarget-v1.yaml",
    "finetune-t1-tabular-regression-missing-v1.yaml",
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

/// Fine-tuning contracts that have been "promoted" to certified — i.e. the
/// implementing PMAT-3NN ticket has landed the recipe + Lean module + binding
/// entries. Only these need to appear in CONTRACT_FILES; the other ~150
/// stubs are auto-generated and exercised by `every_finetune_stub_parses`.
const FINETUNE_CERTIFIED: &[&str] = &[
    // Tier 1.1 SFT minimal × 5 (PMAT-331)
    "finetune-t1-sft-minimal-llama-v1.yaml",
    "finetune-t1-sft-minimal-mistral-v1.yaml",
    "finetune-t1-sft-minimal-phi-v1.yaml",
    "finetune-t1-sft-minimal-qwen-v1.yaml",
    "finetune-t1-sft-minimal-gemma-v1.yaml",
    // Tier 1.2 Eval primitives × 5 (PMAT-332)
    "finetune-t1-eval-perplexity-v1.yaml",
    "finetune-t1-eval-accuracy-v1.yaml",
    "finetune-t1-eval-f1-v1.yaml",
    "finetune-t1-eval-rouge-l-v1.yaml",
    "finetune-t1-eval-bleu-v1.yaml",
    // Tier 1.3 Tabular regression × 5 (PMAT-333)
    "finetune-t1-tabular-regression-housing-v1.yaml",
    "finetune-t1-tabular-regression-energy-v1.yaml",
    "finetune-t1-tabular-regression-timeseries-v1.yaml",
    "finetune-t1-tabular-regression-multitarget-v1.yaml",
    "finetune-t1-tabular-regression-missing-v1.yaml",
];

fn is_finetune_certified(name: &str) -> bool {
    FINETUNE_CERTIFIED.contains(&name)
}

/// Every fine-tuning stub on disk parses as valid YAML. PMAT-330 generated
/// 155 stubs; this test exercises all of them so a generator regression
/// surfaces immediately.
#[test]
fn every_finetune_stub_parses() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    let mut count = 0;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read contracts/").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("finetune-")
            || Path::new(&name).extension().is_none_or(|e| e != "yaml")
        {
            continue;
        }
        count += 1;
        let path = entry.path();
        if let Err(e) = parse_contract(&path) {
            failures.push(format!("{name}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "fine-tuning stub parse failures ({count} stubs):\n  {}",
        failures.join("\n  ")
    );
    assert!(
        count >= 155,
        "expected ≥155 fine-tuning stubs on disk, found {count}"
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
        .filter(|n| {
            // Fine-tuning-cookbook contract stubs (PMAT-330) are auto-generated
            // from manifest.yaml by `scripts/finetune-gen.sh --update`. They are
            // status: planned until the implementing PMAT-3NN ticket lands the
            // recipe; only certified ones go in CONTRACT_FILES. A separate test
            // (`every_finetune_stub_parses` below) exercises them all.
            !n.starts_with("finetune-") || is_finetune_certified(n)
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
