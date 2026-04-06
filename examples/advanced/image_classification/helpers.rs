#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

use proptest::prelude::*;
#[allow(unused_imports)]
use std::f32::consts::PI;

pub fn generate_test_image(seed: u64) -> Result<RgbImage> {
    let mut rng = SimpleRng::new(seed);
    let mut pixels = vec![0.0_f32; NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE];
    for c in 0..NUM_CHANNELS {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let base = match c {
                    0 => x as f32 / IMAGE_SIZE as f32,
                    1 => y as f32 / IMAGE_SIZE as f32,
                    _ => 0.5,
                };
                pixels[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                    (base + rng.next_f32() * 0.1).clamp(0.0, 1.0);
            }
        }
    }
    RgbImage::new(pixels, IMAGE_SIZE, IMAGE_SIZE)
}
