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

pub fn stamp_pattern(pixels: &mut [f32], pattern: &[u8], pw: usize, ph: usize) {
    let (ox, oy) = ((IMAGE_WIDTH - pw) / 2, (IMAGE_HEIGHT - ph) / 2);
    for py in 0..ph {
        for px in 0..pw {
            if px + py * pw < pattern.len() && pattern[py * pw + px] == 1 {
                let (x, y) = (ox + px, oy + py);
                if x < IMAGE_WIDTH && y < IMAGE_HEIGHT {
                    pixels[y * IMAGE_WIDTH + x] = 1.0;
                }
            }
        }
    }
}

pub fn generate_test_digit(digit: u8, seed: u64) -> Result<GrayscaleImage> {
    if digit > 9 {
        return Err(RecognitionError::PreprocessingError(format!(
            "Invalid digit: {digit}"
        )));
    }
    let patterns: [&[u8]; 10] = [
        &[
            0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
            0,
        ],
        &[
            0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
            0,
        ],
        &[
            0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
            0,
        ],
        &[
            1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
            0,
        ],
        &[
            1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            0,
        ],
        &[
            1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
            0,
        ],
        &[
            0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,
            0,
        ],
        &[
            1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0,
        ],
        &[
            0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,
            0,
        ],
        &[
            0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
            0,
        ],
    ];
    let mut pixels = vec![0.0_f32; IMAGE_SIZE];
    stamp_pattern(&mut pixels, patterns[digit as usize], 5, 6);
    let mut rng = SimpleRng::new(seed);
    for p in &mut pixels {
        if *p > 0.5 {
            *p = (*p + rng.next_f32() * 0.1).min(1.0);
        }
    }
    GrayscaleImage::new(pixels, IMAGE_WIDTH, IMAGE_HEIGHT)
}
