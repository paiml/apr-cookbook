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

// --- ImagePreprocessor impl ---

impl ImagePreprocessor {
    pub fn process(&self, image: &RgbImage) -> Result<RgbImage> {
        let rw = (self.target_width as f32 * self.crop_ratio) as usize;
        let rh = (self.target_height as f32 * self.crop_ratio) as usize;
        let resized = self.resize(image, rw, rh)?;
        let cropped = self.center_crop(&resized)?;
        if self.normalize {
            self.apply_imagenet_normalization(&cropped)
        } else {
            Ok(cropped)
        }
    }
    pub fn resize(&self, image: &RgbImage, nw: usize, nh: usize) -> Result<RgbImage> {
        let mut pixels = vec![0.0_f32; NUM_CHANNELS * nh * nw];
        let (sx, sy) = (
            image.width as f32 / nw as f32,
            image.height as f32 / nh as f32,
        );
        for c in 0..NUM_CHANNELS {
            for y in 0..nh {
                for x in 0..nw {
                    let (fx, fy) = (x as f32 * sx, y as f32 * sy);
                    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                    let (x1, y1) = (
                        (x0 + 1).min(image.width - 1),
                        (y0 + 1).min(image.height - 1),
                    );
                    let (dx, dy) = (fx - x0 as f32, fy - y0 as f32);
                    let p = |yy, xx| image.get_pixel(c, yy, xx).unwrap_or(0.0);
                    pixels[c * nh * nw + y * nw + x] = p(y0, x0) * (1.0 - dx) * (1.0 - dy)
                        + p(y0, x1) * dx * (1.0 - dy)
                        + p(y1, x0) * (1.0 - dx) * dy
                        + p(y1, x1) * dx * dy;
                }
            }
        }
        RgbImage::new(pixels, nw, nh)
    }
    pub fn center_crop(&self, image: &RgbImage) -> Result<RgbImage> {
        let (ox, oy) = (
            (image.width.saturating_sub(self.target_width)) / 2,
            (image.height.saturating_sub(self.target_height)) / 2,
        );
        let (tw, th) = (self.target_width, self.target_height);
        let mut pixels = vec![0.0_f32; NUM_CHANNELS * th * tw];
        for c in 0..NUM_CHANNELS {
            for y in 0..th {
                for x in 0..tw {
                    pixels[c * th * tw + y * tw + x] =
                        image.get_pixel(c, y + oy, x + ox).unwrap_or(0.0);
                }
            }
        }
        RgbImage::new(pixels, tw, th)
    }
    pub fn apply_imagenet_normalization(&self, image: &RgbImage) -> Result<RgbImage> {
        let mut pixels = image.pixels.clone();
        for c in 0..NUM_CHANNELS {
            for i in 0..(image.height * image.width) {
                let idx = c * image.height * image.width + i;
                pixels[idx] = (pixels[idx] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
        RgbImage::new(pixels, image.width, image.height)
    }
}

// --- ImageClassifier impl ---

impl ImageClassifier {
    pub fn predict(&self, image: &RgbImage) -> Result<ClassificationResult> {
        if image.width != IMAGE_SIZE || image.height != IMAGE_SIZE {
            return Err(ClassificationError::InvalidDimensions {
                expected_width: IMAGE_SIZE,
                expected_height: IMAGE_SIZE,
                got_width: image.width,
                got_height: image.height,
            });
        }
        let stem_out = self.stem_conv(&image.pixels);
        let pooled = self.global_avg_pool(&stem_out, 112, 112, 16);
        let mut features = vec![0.0_f32; self.weights.feature_dim];
        for (i, &v) in pooled.iter().enumerate() {
            if i < features.len() {
                features[i] = v;
            }
        }
        let logits = self.dense(
            &features,
            &self.weights.classifier_weights,
            &self.weights.classifier_bias,
        );
        let probabilities = self.softmax(&logits);
        let (pc, conf) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, 0.0), |(i, &p)| (i, p));
        Ok(ClassificationResult {
            predicted_class: pc,
            label: self.labels.get(pc).cloned().unwrap_or_default(),
            confidence: conf,
            probabilities,
        })
    }
    pub fn predict_batch(&self, images: &[RgbImage]) -> Result<Vec<ClassificationResult>> {
        images.iter().map(|img| self.predict(img)).collect()
    }
    pub fn stem_conv(&self, input: &[f32]) -> Vec<f32> {
        let (ih, iw, ic, oc, k, s) = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS, 16, 3, 2);
        let (oh, ow) = ((ih - k) / s + 1, (iw - k) / s + 1);
        let mut out = vec![0.0_f32; oc * oh * ow];
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = self.weights.stem_bias[o];
                    for c in 0..ic {
                        for ky in 0..k {
                            for kx in 0..k {
                                let (iy, ix) = (oy * s + ky, ox * s + kx);
                                if iy < ih && ix < iw {
                                    let ii = c * ih * iw + iy * iw + ix;
                                    let wi = ((o * ic + c) * k + ky) * k + kx;
                                    if ii < input.len() && wi < self.weights.stem_weights.len() {
                                        sum += input[ii] * self.weights.stem_weights[wi];
                                    }
                                }
                            }
                        }
                    }
                    let oi = o * oh * ow + oy * ow + ox;
                    if oi < out.len() {
                        out[oi] = Activation::HardSwish.apply(sum);
                    }
                }
            }
        }
        out
    }
    pub fn global_avg_pool(&self, input: &[f32], h: usize, w: usize, ch: usize) -> Vec<f32> {
        let s = h * w;
        (0..ch)
            .map(|c| {
                (0..s)
                    .filter_map(|i| input.get(c * s + i).copied())
                    .sum::<f32>()
                    / s as f32
            })
            .collect()
    }
    pub fn dense(&self, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut out: Vec<f32> = bias.to_vec();
        for (o, v) in out.iter_mut().enumerate() {
            for (i, &iv) in input.iter().enumerate() {
                let wi = o * n + i;
                if wi < weights.len() {
                    *v += iv * weights[wi];
                }
            }
        }
        out
    }
    pub fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let es: f32 = logits.iter().map(|&x| (x - mx).exp()).sum();
        logits.iter().map(|&x| (x - mx).exp() / es).collect()
    }
}
