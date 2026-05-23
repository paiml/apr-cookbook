//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
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

// --- ImagePreprocessor impl ---

impl ImagePreprocessor {
    pub fn process(&self, image: &GrayscaleImage) -> Result<GrayscaleImage> {
        let mut r = image.clone();
        if self.invert {
            for p in &mut r.pixels {
                *p = 1.0 - *p;
            }
        }
        if self.denoise > 0.0 {
            let th = self.denoise * 0.1;
            r = GrayscaleImage::new(
                r.pixels
                    .iter()
                    .map(|&p| if p < th { 0.0 } else { p })
                    .collect(),
                r.width,
                r.height,
            )?;
        }
        if self.center {
            r = self.center_digit(&r)?;
        }
        Ok(r)
    }
    pub fn center_digit(&self, img: &GrayscaleImage) -> Result<GrayscaleImage> {
        let (mut cx, mut cy, mut total) = (0.0_f64, 0.0_f64, 0.0_f64);
        for y in 0..img.height {
            for x in 0..img.width {
                let p = f64::from(img.pixels[y * img.width + x]);
                cx += p * x as f64;
                cy += p * y as f64;
                total += p;
            }
        }
        if total < 1e-6 {
            return Ok(img.clone());
        }
        let (sx, sy) = (
            (img.width as f64 / 2.0 - cx / total).round() as i32,
            (img.height as f64 / 2.0 - cy / total).round() as i32,
        );
        let mut pixels = vec![0.0_f32; img.pixels.len()];
        for y in 0..img.height {
            for x in 0..img.width {
                let (nx, ny) = (x as i32 - sx, y as i32 - sy);
                if nx >= 0 && nx < img.width as i32 && ny >= 0 && ny < img.height as i32 {
                    pixels[y * img.width + x] = img.pixels[ny as usize * img.width + nx as usize];
                }
            }
        }
        GrayscaleImage::new(pixels, img.width, img.height)
    }
}

// --- ImageAugmenter impl ---

impl ImageAugmenter {
    pub fn rotate(&self, img: &GrayscaleImage, degrees: f32) -> Result<GrayscaleImage> {
        let (cos_a, sin_a) = ((degrees * PI / 180.0).cos(), (degrees * PI / 180.0).sin());
        let (cx, cy) = (img.width as f32 / 2.0, img.height as f32 / 2.0);
        let mut pixels = vec![0.0_f32; img.pixels.len()];
        for y in 0..img.height {
            for x in 0..img.width {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if let Some(v) = self.bilinear_sample(
                    img,
                    dx * cos_a + dy * sin_a + cx,
                    -dx * sin_a + dy * cos_a + cy,
                ) {
                    pixels[y * img.width + x] = v;
                }
            }
        }
        GrayscaleImage::new(pixels, img.width, img.height)
    }
    pub fn scale(&self, img: &GrayscaleImage, factor: f32) -> Result<GrayscaleImage> {
        let (cx, cy) = (img.width as f32 / 2.0, img.height as f32 / 2.0);
        let mut pixels = vec![0.0_f32; img.pixels.len()];
        for y in 0..img.height {
            for x in 0..img.width {
                if let Some(v) = self.bilinear_sample(
                    img,
                    (x as f32 - cx) / factor + cx,
                    (y as f32 - cy) / factor + cy,
                ) {
                    pixels[y * img.width + x] = v;
                }
            }
        }
        GrayscaleImage::new(pixels, img.width, img.height)
    }
    pub fn add_noise(&mut self, img: &GrayscaleImage, sigma: f32) -> Result<GrayscaleImage> {
        let pixels: Vec<f32> = img
            .pixels
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let h = {
                    let mut h = self.seed.wrapping_add(i as u64);
                    h ^= h >> 33;
                    h = h.wrapping_mul(0xff51afd7ed558ccd);
                    h ^= h >> 33;
                    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
                    h ^= h >> 33;
                    h
                };
                let (u1, u2) = (
                    (h & 0xFFFF) as f32 / 65535.0_f32,
                    ((h >> 16) & 0xFFFF) as f32 / 65535.0,
                );
                (p + (-2.0 * u1.max(1e-6).ln()).sqrt() * (2.0 * PI * u2).cos() * sigma)
                    .clamp(0.0, 1.0)
            })
            .collect();
        GrayscaleImage::new(pixels, img.width, img.height)
    }
    pub fn bilinear_sample(&self, img: &GrayscaleImage, x: f32, y: f32) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x >= img.width as f32 - 1.0 || y >= img.height as f32 - 1.0 {
            return Some(0.0);
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        Some(
            img.get_pixel(x0, y0).unwrap_or(0.0) * (1.0 - fx) * (1.0 - fy)
                + img.get_pixel(x0 + 1, y0).unwrap_or(0.0) * fx * (1.0 - fy)
                + img.get_pixel(x0, y0 + 1).unwrap_or(0.0) * (1.0 - fx) * fy
                + img.get_pixel(x0 + 1, y0 + 1).unwrap_or(0.0) * fx * fy,
        )
    }
}

// --- LeNetWeights impl ---

impl LeNetWeights {
    pub fn random_init(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let g = |n: usize, scale: f32, rng: &mut SimpleRng| -> Vec<f32> {
            (0..n).map(|_| rng.next_gaussian() * scale).collect()
        };
        Self {
            conv1_weights: g(150, (2.0 / 25.0_f32).sqrt(), &mut rng),
            conv1_bias: vec![0.0; 6],
            conv2_weights: g(2400, (2.0 / 150.0_f32).sqrt(), &mut rng),
            conv2_bias: vec![0.0; 16],
            fc1_weights: g(30720, (2.0 / 256.0_f32).sqrt(), &mut rng),
            fc1_bias: vec![0.0; 120],
            fc2_weights: g(10080, (2.0 / 120.0_f32).sqrt(), &mut rng),
            fc2_bias: vec![0.0; 84],
            fc3_weights: g(840, (2.0 / 84.0_f32).sqrt(), &mut rng),
            fc3_bias: vec![0.0; 10],
        }
    }
    pub fn validate(&self) -> Result<()> {
        let expected = [
            (self.conv1_weights.len(), 150),
            (self.conv1_bias.len(), 6),
            (self.conv2_weights.len(), 2400),
            (self.conv2_bias.len(), 16),
            (self.fc1_weights.len(), 30720),
            (self.fc1_bias.len(), 120),
            (self.fc2_weights.len(), 10080),
            (self.fc2_bias.len(), 84),
            (self.fc3_weights.len(), 840),
            (self.fc3_bias.len(), 10),
        ];
        for (got, exp) in expected {
            if got != exp {
                return Err(RecognitionError::WeightError(format!(
                    "expected {exp}, got {got}"
                )));
            }
        }
        for &w in self
            .conv1_weights
            .iter()
            .chain(&self.conv1_bias)
            .chain(&self.conv2_weights)
            .chain(&self.conv2_bias)
            .chain(&self.fc1_weights)
            .chain(&self.fc1_bias)
            .chain(&self.fc2_weights)
            .chain(&self.fc2_bias)
            .chain(&self.fc3_weights)
            .chain(&self.fc3_bias)
        {
            if w.is_nan() || w.is_infinite() {
                return Err(RecognitionError::WeightError(format!(
                    "Invalid weight: {w}"
                )));
            }
        }
        Ok(())
    }
}

// --- LeNetClassifier impl ---

impl LeNetClassifier {
    pub fn predict(&self, image: &GrayscaleImage) -> Result<Prediction> {
        if image.pixels.len() != IMAGE_SIZE {
            return Err(RecognitionError::InvalidDimensions {
                expected: IMAGE_SIZE,
                got: image.pixels.len(),
            });
        }
        let c1 = self.conv2d(
            &image.pixels,
            28,
            28,
            1,
            &self.weights.conv1_weights,
            &self.weights.conv1_bias,
            6,
            5,
        );
        let p1 = self.max_pool(&c1, 24, 24, 6);
        let c2 = self.conv2d(
            &p1,
            12,
            12,
            6,
            &self.weights.conv2_weights,
            &self.weights.conv2_bias,
            16,
            5,
        );
        let p2 = self.max_pool(&c2, 8, 8, 16);
        let f1 = self.dense(&p2, &self.weights.fc1_weights, &self.weights.fc1_bias, true);
        let f2 = self.dense(&f1, &self.weights.fc2_weights, &self.weights.fc2_bias, true);
        let logits = self.dense(
            &f2,
            &self.weights.fc3_weights,
            &self.weights.fc3_bias,
            false,
        );
        let probs = self.softmax(&logits);
        let (cls, conf) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, 0.0), |(i, &p)| (i, p));
        Ok(Prediction {
            predicted_class: cls,
            confidence: conf,
            probabilities: probs,
        })
    }
    #[allow(clippy::needless_range_loop)]
    pub fn conv2d(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        ic: usize,
        weights: &[f32],
        bias: &[f32],
        oc: usize,
        k: usize,
    ) -> Vec<f32> {
        let (oh, ow) = (h - k + 1, w - k + 1);
        let mut out = vec![0.0_f32; oh * ow * oc];
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = bias[o];
                    for c in 0..ic {
                        for ky in 0..k {
                            for kx in 0..k {
                                let ii = (c * h + oy + ky) * w + ox + kx;
                                let wi = ((o * ic + c) * k + ky) * k + kx;
                                if ii < input.len() && wi < weights.len() {
                                    sum += input[ii] * weights[wi];
                                }
                            }
                        }
                    }
                    let idx = (o * oh + oy) * ow + ox;
                    if idx < out.len() {
                        out[idx] = sum.max(0.0);
                    }
                }
            }
        }
        out
    }
    pub fn max_pool(&self, input: &[f32], h: usize, w: usize, ch: usize) -> Vec<f32> {
        let (oh, ow) = (h / 2, w / 2);
        let mut out = vec![0.0_f32; oh * ow * ch];
        for c in 0..ch {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut mx = f32::NEG_INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let i = (c * h + oy * 2 + dy) * w + ox * 2 + dx;
                            if i < input.len() {
                                mx = mx.max(input[i]);
                            }
                        }
                    }
                    let idx = (c * oh + oy) * ow + ox;
                    if idx < out.len() {
                        out[idx] = mx;
                    }
                }
            }
        }
        out
    }
    pub fn dense(&self, input: &[f32], weights: &[f32], bias: &[f32], relu: bool) -> Vec<f32> {
        let mut out: Vec<f32> = bias.to_vec();
        for (o, val) in out.iter_mut().enumerate() {
            for (i, &iv) in input.iter().enumerate() {
                let wi = o * input.len() + i;
                if wi < weights.len() {
                    *val += iv * weights[wi];
                }
            }
            if relu {
                *val = val.max(0.0);
            }
        }
        out
    }
    pub fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - mx).exp()).sum();
        logits.iter().map(|&x| (x - mx).exp() / sum).collect()
    }
}
