//! Utility to generate `assets/demo_model.apr` — a small, deterministic APR v1
//! model file suitable for use in tests, documentation, and bundled-model examples.
//!
//! ## Run Command
//! ```bash
//! cargo run --example create_demo_model
//! ```
//!
//! ## References
//! - Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR. arXiv:1712.05877

use apr_cookbook::bundle::ModelBundle;
use std::path::Path;

fn main() {
    // Deterministic 128-weight payload (512 bytes of f32 LE).
    // Values form a simple linear ramp so the file is fully reproducible.
    let n_weights: usize = 128;
    let payload: Vec<u8> = (0..n_weights)
        .flat_map(|i| {
            let val = (i as f32) / (n_weights as f32);
            val.to_le_bytes()
        })
        .collect();

    let apr_bytes = ModelBundle::new()
        .with_name("demo-model")
        .with_description("Deterministic demo model for cookbook assets")
        .with_payload(payload)
        .build();

    let dest = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo_model.apr");
    std::fs::write(&dest, &apr_bytes).expect("failed to write demo_model.apr");

    println!("Wrote {} bytes to {}", apr_bytes.len(), dest.display());
}
