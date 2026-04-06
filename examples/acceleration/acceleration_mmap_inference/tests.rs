#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

/// Helper: build a deterministic recipe context for tests.
fn test_ctx() -> RecipeContext {
    RecipeContext::new("test_mmap_inference").expect("context creation")
}

#[test]
fn test_create_synthetic_model_produces_correct_count() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    assert_eq!(tensors.len(), NUM_TENSORS);
}

#[test]
fn test_each_tensor_has_correct_element_count() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    for t in &tensors {
        assert_eq!(t.data.len(), ELEMENTS_PER_TENSOR);
    }
}

#[test]
fn test_write_and_read_roundtrip() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let path = ctx.path("roundtrip.bin");
    write_model_file(&path, &tensors).expect("write");

    let raw = std::fs::read(&path).expect("read");
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let parsed = parse_tensors_from_bytes(&raw, &names);

    for (orig, parsed) in tensors.iter().zip(parsed.iter()) {
        assert_eq!(orig.name, parsed.name);
        assert_eq!(orig.data.len(), parsed.data.len());
        for (a, b) in orig.data.iter().zip(parsed.data.iter()) {
            assert!((a - b).abs() < 1e-15, "mismatch: {a} vs {b}");
        }
    }
}

#[test]
fn test_forward_pass_output_size() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let output = forward_pass(&tensors);
    assert_eq!(output.len(), 16);
}

#[test]
fn test_forward_pass_values_bounded() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let output = forward_pass(&tensors);
    // tanh output is always in (-1, 1)
    for &v in &output {
        assert!(v.abs() < 1.0, "tanh output must be in (-1,1), got {v}");
    }
}

#[test]
fn test_mmap_view_tracks_page_access() {
    let data = vec![0u8; PAGE_SIZE * 4];
    let mut view = MmapView::new(data);
    assert_eq!(view.resident_pages(), 0);

    // Access page 1 only
    let _ = view.read_range(PAGE_SIZE, 8);
    assert_eq!(view.resident_pages(), 1);
    assert!(!view.page_accessed[0]);
    assert!(view.page_accessed[1]);
}

#[test]
fn test_mmap_forward_pass_matches_eager() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let path = ctx.path("match.bin");
    write_model_file(&path, &tensors).expect("write");

    let raw = std::fs::read(&path).expect("read");
    let mut view = MmapView::new(raw);

    let eager_output = forward_pass(&tensors);
    let mmap_output = forward_pass_mmap(&mut view);

    assert_eq!(eager_output.len(), mmap_output.len());
    for (a, b) in eager_output.iter().zip(mmap_output.iter()) {
        assert!((a - b).abs() < 1e-10, "eager vs mmap mismatch: {a} vs {b}");
    }
}

#[test]
fn test_mmap_skips_unused_tensor_pages() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let path = ctx.path("skip.bin");
    write_model_file(&path, &tensors).expect("write");

    let raw = std::fs::read(&path).expect("read");
    let mut view = MmapView::new(raw);
    let _ = forward_pass_mmap(&mut view);

    // Tensor 3 (the last one) should NOT have any pages accessed
    let bytes_per_tensor = ELEMENTS_PER_TENSOR * 8;
    let tensor3_first_page = (3 * bytes_per_tensor) / PAGE_SIZE;
    for p in tensor3_first_page..view.page_count {
        assert!(
            !view.page_accessed[p],
            "page {p} in tensor 3 should not be accessed"
        );
    }
}

#[test]
fn test_load_strategy_display() {
    assert_eq!(format!("{}", LoadStrategy::Eager), "Eager");
    assert_eq!(format!("{}", LoadStrategy::MemoryMapped), "MemoryMapped");
}

#[test]
fn test_page_report_annotations() {
    let mut ctx = test_ctx();
    let tensors = create_synthetic_model(ctx.rng());
    let path = ctx.path("report.bin");
    write_model_file(&path, &tensors).expect("write");

    let raw = std::fs::read(&path).expect("read");
    let mut view = MmapView::new(raw);
    let _ = forward_pass_mmap(&mut view);
    let report = view.page_report(&tensors);

    assert!(!report.is_empty());
    // First page should belong to layer0.weight
    assert_eq!(report[0].tensor_name, "layer0.weight");
    // Every page should have a non-empty tensor name
    for page in &report {
        assert!(!page.tensor_name.is_empty());
    }
}
