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

criterion_group!(
    benches,
    benchmark_bundled_model_parsing,
    benchmark_model_bundle_creation
);
criterion_main!(benches);
