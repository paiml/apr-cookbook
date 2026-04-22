//! Deterministic test-data generators used by recipes.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Hash a recipe name to a deterministic u64 seed.
///
/// Uses BLAKE3 for consistent cross-platform hashing.
#[must_use]
pub fn hash_name_to_seed(name: &str) -> u64 {
    let hash = blake3::hash(name.as_bytes());
    let bytes = hash.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Generate deterministic test data for a given seed.
///
/// Useful for creating reproducible test fixtures.
#[must_use]
pub fn generate_test_data(seed: u64, size: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generate a deterministic model payload for testing.
///
/// Creates fake "model weights" that are reproducible.
#[must_use]
pub fn generate_model_payload(seed: u64, n_params: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let weights: Vec<f32> = (0..n_params)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    // Serialize as raw f32 bytes
    weights.iter().flat_map(|f| f.to_le_bytes()).collect()
}
