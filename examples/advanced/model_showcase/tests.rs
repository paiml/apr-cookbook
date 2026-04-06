#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

#[test]
fn test_full_pipeline_succeeds() {
    let mut ctx = RecipeContext::new("test_showcase_full").expect("context");
    let report = run_showcase_pipeline(&mut ctx).expect("pipeline");
    assert_eq!(report.steps.len(), 6);
    assert_eq!(report.fail_count(), 0);
    assert_eq!(report.done_count(), 6);
}

#[test]
fn test_step_create_produces_valid_bundle() {
    let mut ctx = RecipeContext::new("test_create").expect("context");
    let (step, created) = step_create(&mut ctx, "test-model").expect("create");
    assert_eq!(step.status, StepStatus::Done);
    assert!(created.bytes.len() > 64);
    assert_eq!(&created.bytes[0..4], b"APR2");
    assert_eq!(created.n_tensors, 4);
}

#[test]
fn test_step_inspect_parses_header() {
    let mut ctx = RecipeContext::new("test_inspect").expect("context");
    let (_step, created) = step_create(&mut ctx, "inspect-model").expect("create");
    let step = step_inspect(&created).expect("inspect");
    assert_eq!(step.status, StepStatus::Done);
    assert!(step.detail.contains("tensors=4"));
}

#[test]
fn test_step_validate_clean_model() {
    let mut ctx = RecipeContext::new("test_validate").expect("context");
    let (_step, created) = step_create(&mut ctx, "validate-model").expect("create");
    let step = step_validate(&created).expect("validate");
    assert_eq!(step.status, StepStatus::Done);
    assert!(step.detail.contains("nan_count=0"));
}

#[test]
fn test_step_validate_detects_bad_magic() {
    let model = CreatedModel {
        bytes: vec![0xFF; 128],
        n_params: 0,
        n_tensors: 0,
    };
    // Bad magic should cause a parse error in BundledModelV2
    let result = step_validate(&model);
    assert!(result.is_err());
}

#[test]
fn test_step_benchmark_reports_throughput() {
    let mut ctx = RecipeContext::new("test_bench").expect("context");
    let (_step, created) = step_create(&mut ctx, "bench-model").expect("create");
    let step = step_benchmark(&mut ctx, &created).expect("benchmark");
    assert_eq!(step.status, StepStatus::Done);
    assert!(step.detail.contains("inferences/sec"));
}

#[test]
fn test_step_convert_rewrites_magic() {
    let mut ctx = RecipeContext::new("test_convert").expect("context");
    let (_step, created) = step_create(&mut ctx, "convert-model").expect("create");
    let (step, converted) = step_convert(&created).expect("convert");
    assert_eq!(step.status, StepStatus::Done);
    assert_eq!(&converted[0..4], b"GGUF");
    // Payload after magic should be identical
    assert_eq!(&created.bytes[4..], &converted[4..]);
}

#[test]
fn test_step_compare_identical_payloads() {
    let mut ctx = RecipeContext::new("test_compare").expect("context");
    let (_step, created) = step_create(&mut ctx, "compare-model").expect("create");
    let (_, converted) = step_convert(&created).expect("convert");
    let step = step_compare(&created.bytes, &converted).expect("compare");
    assert_eq!(step.status, StepStatus::Done);
    assert!(step.detail.contains("payload_identical=true"));
}

#[test]
fn test_float_roundtrip() {
    let original: Vec<f32> = vec![1.0, -0.5, 0.0, f32::MAX, f32::MIN_POSITIVE];
    let bytes = float_vec_to_bytes(&original);
    let recovered = bytes_to_float_vec(&bytes);
    assert_eq!(original, recovered);
}

#[test]
fn test_pipeline_deterministic() {
    let mut ctx1 = RecipeContext::new("test_determinism").expect("ctx1");
    let mut ctx2 = RecipeContext::new("test_determinism").expect("ctx2");
    let r1 = run_showcase_pipeline(&mut ctx1).expect("r1");
    let r2 = run_showcase_pipeline(&mut ctx2).expect("r2");
    assert_eq!(r1.steps.len(), r2.steps.len());
    for (a, b) in r1.steps.iter().zip(r2.steps.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.status, b.status);
    }
}
