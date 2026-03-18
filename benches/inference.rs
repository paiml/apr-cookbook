//! Inference performance benchmarks.

#![allow(clippy::disallowed_methods)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_bundled_model_parsing(c: &mut Criterion) {
    use apr_cookbook::bundle::ModelBundle;

    // Create a test model bundle
    let bundle = ModelBundle::new()
        .with_name("benchmark-model")
        .with_payload(vec![0u8; 10000])
        .build();

    c.bench_function("parse_bundled_model", |b| {
        b.iter(|| {
            let model = apr_cookbook::bundle::BundledModel::from_bytes(black_box(&bundle)).unwrap();
            black_box(model.size())
        });
    });
}

fn benchmark_model_bundle_creation(c: &mut Criterion) {
    use apr_cookbook::bundle::ModelBundle;

    let payload = vec![0u8; 10000];

    c.bench_function("create_model_bundle", |b| {
        b.iter(|| {
            let bundle = ModelBundle::new()
                .with_name(black_box("test-model"))
                .with_payload(black_box(payload.clone()))
                .build();
            black_box(bundle.len())
        });
    });
}

fn benchmark_lz4_compression(c: &mut Criterion) {
    // Generate deterministic 1MB of pseudo-random data
    let data: Vec<u8> = (0..1_048_576u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
        .collect();

    c.bench_function("lz4_compress_decompress_1mb", |b| {
        b.iter(|| {
            let compressed = lz4_flex::compress_prepend_size(black_box(&data));
            let decompressed = lz4_flex::decompress_size_prepended(black_box(&compressed)).unwrap();
            black_box(decompressed.len())
        });
    });
}

fn benchmark_zstd_compression(c: &mut Criterion) {
    // Generate deterministic 1MB of pseudo-random data
    let data: Vec<u8> = (0..1_048_576u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
        .collect();

    c.bench_function("zstd_compress_decompress_1mb", |b| {
        b.iter(|| {
            let compressed = zstd::encode_all(std::io::Cursor::new(black_box(&data)), 3).unwrap();
            let decompressed =
                zstd::decode_all(std::io::Cursor::new(black_box(&compressed))).unwrap();
            black_box(decompressed.len())
        });
    });
}

fn benchmark_lz4_decompression_throughput(c: &mut Criterion) {
    // Generate deterministic 1MB of pseudo-random data and pre-compress
    let data: Vec<u8> = (0..1_048_576u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    let compressed = lz4_flex::compress_prepend_size(&data);

    c.bench_function("lz4_decompress_only_1mb", |b| {
        b.iter(|| {
            let decompressed = lz4_flex::decompress_size_prepended(black_box(&compressed)).unwrap();
            black_box(decompressed.len())
        });
    });
}

fn benchmark_matmul_trueno(c: &mut Criterion) {
    use trueno::Matrix;

    let mut group = c.benchmark_group("matmul_trueno");
    for &size in &[64, 128, 256, 512] {
        // Deterministic data: value = ((row * cols + col) * prime) mod 256, scaled to f32
        let n = size * size;
        let a_data: Vec<f32> = (0..n)
            .map(|i| ((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0)
            .collect();
        let b_data: Vec<f32> = (0..n)
            .map(|i| ((i as u64).wrapping_mul(2246822519) % 1000) as f32 / 1000.0)
            .collect();

        let a = Matrix::from_vec(size, size, a_data).expect("Failed to create matrix A");
        let b = Matrix::from_vec(size, size, b_data).expect("Failed to create matrix B");

        group.bench_function(format!("{size}x{size}"), |bench| {
            bench.iter(|| {
                let result = black_box(&a).matmul(black_box(&b));
                black_box(result)
            });
        });
    }
    group.finish();
}

fn benchmark_matmul_scalar(c: &mut Criterion) {
    let size = 256;
    let n = size * size;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0)
        .collect();
    let b: Vec<f32> = (0..n)
        .map(|i| ((i as u64).wrapping_mul(2246822519) % 1000) as f32 / 1000.0)
        .collect();

    c.bench_function("matmul_scalar_256x256", |bench| {
        bench.iter(|| {
            let mut c_out = vec![0.0f32; size * size];
            for i in 0..size {
                for k in 0..size {
                    let a_ik = black_box(a[i * size + k]);
                    for j in 0..size {
                        c_out[i * size + j] += a_ik * black_box(b[k * size + j]);
                    }
                }
            }
            black_box(c_out)
        });
    });
}

criterion_group!(
    benches,
    benchmark_bundled_model_parsing,
    benchmark_model_bundle_creation,
    benchmark_lz4_compression,
    benchmark_zstd_compression,
    benchmark_lz4_decompression_throughput,
    benchmark_matmul_trueno,
    benchmark_matmul_scalar
);
criterion_main!(benches);
