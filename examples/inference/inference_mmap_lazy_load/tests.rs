#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

#[test]
fn test_create_model_produces_valid_file() {
    let ctx = RecipeContext::new("test_create_model").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 1, &mut rng).expect("create");
    assert!(!tensors.is_empty());
    assert!(path.exists());
}

#[test]
fn test_create_model_file_size() {
    let ctx = RecipeContext::new("test_file_size").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let _tensors = create_test_model(&path, 1, &mut rng).expect("create");
    let meta = std::fs::metadata(&path).expect("meta");
    // 1 MB of data plus header + tensor metadata
    assert!(meta.len() >= 1024 * 1024);
}

#[test]
fn test_simulate_mmap_load_validates_tensors() {
    let ctx = RecipeContext::new("test_mmap_validate").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 1, &mut rng).expect("create");
    let loader = simulate_mmap_load(&path, &tensors).expect("mmap");
    assert!(loader.file_size >= 1024 * 1024);
    assert!(loader.mapped_regions.is_empty());
}

#[test]
fn test_lazy_load_single_tensor() {
    let ctx = RecipeContext::new("test_lazy_single").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 1, &mut rng).expect("create");
    let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");

    let data = lazy_load_tensor(&mut loader, &tensors[0]).expect("load");
    assert_eq!(data.len(), 65_536); // 256 KiB / 4 bytes per float
    assert_eq!(loader.mapped_regions.len(), 1);
    assert_eq!(loader.total_bytes_read, tensors[0].length);
}

#[test]
fn test_lazy_load_tracks_access_count() {
    let ctx = RecipeContext::new("test_access_count").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 1, &mut rng).expect("create");
    let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");

    let _ = lazy_load_tensor(&mut loader, &tensors[0]).expect("load1");
    let _ = lazy_load_tensor(&mut loader, &tensors[0]).expect("load2");

    assert_eq!(loader.mapped_regions.len(), 1);
    assert_eq!(loader.mapped_regions[0].access_count, 2);
    // total_bytes_read only increments on first access
    assert_eq!(loader.total_bytes_read, tensors[0].length);
}

#[test]
fn test_eager_load_reads_entire_file() {
    let ctx = RecipeContext::new("test_eager").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let _tensors = create_test_model(&path, 1, &mut rng).expect("create");

    let (dur, bytes) = benchmark_eager_load(&path).expect("eager");
    let file_len = std::fs::metadata(&path).expect("meta").len() as usize;
    assert_eq!(bytes, file_len);
    assert!(dur.as_nanos() > 0);
}

#[test]
fn test_lazy_load_reads_less_than_eager() {
    let ctx = RecipeContext::new("test_lazy_less").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 2, &mut rng).expect("create");

    let (_, eager_bytes) = benchmark_eager_load(&path).expect("eager");

    // Load 20% of tensors
    let n = (tensors.len() / 5).max(1);
    let indices: Vec<usize> = (0..n).collect();
    let (_, lazy_bytes) = benchmark_lazy_load(&path, &tensors, &indices).expect("lazy");

    assert!(
        lazy_bytes < eager_bytes,
        "lazy {} should be < eager {}",
        lazy_bytes,
        eager_bytes
    );
}

#[test]
fn test_tensor_data_deterministic() {
    let ctx = RecipeContext::new("test_deterministic").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(99);
    let tensors = create_test_model(&path, 1, &mut rng).expect("create");

    let mut loader1 = simulate_mmap_load(&path, &tensors).expect("mmap1");
    let data1 = lazy_load_tensor(&mut loader1, &tensors[0]).expect("load1");

    let mut loader2 = simulate_mmap_load(&path, &tensors).expect("mmap2");
    let data2 = lazy_load_tensor(&mut loader2, &tensors[0]).expect("load2");

    assert_eq!(data1, data2);
}

#[test]
fn test_invalid_model_size_rejected() {
    let ctx = RecipeContext::new("test_invalid_size").expect("ctx");
    let path = ctx.path("tiny.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    // 0 MB should fail
    let result = create_test_model(&path, 0, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_mmap_rejects_truncated_file() {
    let ctx = RecipeContext::new("test_truncated").expect("ctx");
    let path = ctx.path("truncated.bin");

    // Write a tiny file
    std::fs::write(&path, b"APRM").expect("write");

    let fake_tensor = ModelTensor {
        name: "fake".to_string(),
        shape: vec![4, 4],
        dtype: "f32".to_string(),
        offset: 1000,
        length: 4096,
    };
    let result = simulate_mmap_load(&path, &[fake_tensor]);
    assert!(result.is_err());
}

#[test]
fn test_multiple_tensors_independent() {
    let ctx = RecipeContext::new("test_multi_tensor").expect("ctx");
    let path = ctx.path("model.bin");
    let mut rng = rand::SeedableRng::seed_from_u64(42);
    let tensors = create_test_model(&path, 2, &mut rng).expect("create");
    assert!(tensors.len() >= 2);

    let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");
    let d0 = lazy_load_tensor(&mut loader, &tensors[0]).expect("t0");
    let d1 = lazy_load_tensor(&mut loader, &tensors[1]).expect("t1");

    // Different tensors should have different data (random fill)
    assert_ne!(d0, d1);
    assert_eq!(loader.mapped_regions.len(), 2);
}
