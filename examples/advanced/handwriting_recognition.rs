//! # Demo I: Handwriting Recognition (MNIST)
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! LeNet-5 CNN for digit recognition: 28x28 grayscale -> 10 classes.
//! Image preprocessing, augmentation, convolutional inference, and evaluation.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of IEEE. DOI: 10.1109/5.726791

use std::f32::consts::PI;

pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;
pub const NUM_CLASSES: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub enum RecognitionError {
    InvalidDimensions { expected: usize, got: usize },
    InvalidPixelValue { index: usize, value: f32 },
    WeightError(String),
    PreprocessingError(String),
    InferenceError(String),
}
impl std::fmt::Display for RecognitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions { expected, got } => {
                write!(f, "Invalid dimensions: expected {expected}, got {got}")
            }
            Self::InvalidPixelValue { index, value } => {
                write!(f, "Invalid pixel at {index}: {value}")
            }
            Self::WeightError(m) => write!(f, "Weight error: {m}"),
            Self::PreprocessingError(m) => write!(f, "Preprocessing error: {m}"),
            Self::InferenceError(m) => write!(f, "Inference error: {m}"),
        }
    }
}
impl std::error::Error for RecognitionError {}
pub type Result<T> = std::result::Result<T, RecognitionError>;

#[derive(Debug, Clone)]
pub struct GrayscaleImage {
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl GrayscaleImage {
    pub fn new(pixels: Vec<f32>, width: usize, height: usize) -> Result<Self> {
        if pixels.len() != width * height {
            return Err(RecognitionError::InvalidDimensions {
                expected: width * height,
                got: pixels.len(),
            });
        }
        for (i, &p) in pixels.iter().enumerate() {
            if p.is_nan() || p.is_infinite() {
                return Err(RecognitionError::InvalidPixelValue { index: i, value: p });
            }
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }
    pub fn from_mnist_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != IMAGE_SIZE {
            return Err(RecognitionError::InvalidDimensions {
                expected: IMAGE_SIZE,
                got: bytes.len(),
            });
        }
        Self::new(
            bytes.iter().map(|&b| f32::from(b) / 255.0).collect(),
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        )
    }
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.pixels[y * self.width + x])
        } else {
            None
        }
    }
    pub fn set_pixel(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = value.clamp(0.0, 1.0);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImagePreprocessor {
    pub invert: bool,
    pub center: bool,
    pub denoise: f32,
}

impl ImagePreprocessor {
    pub fn new() -> Self {
        Self {
            invert: false,
            center: true,
            denoise: 0.0,
        }
    }
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
    fn center_digit(&self, img: &GrayscaleImage) -> Result<GrayscaleImage> {
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

#[derive(Debug, Clone)]
pub struct ImageAugmenter {
    seed: u64,
}

impl ImageAugmenter {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
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
    fn bilinear_sample(&self, img: &GrayscaleImage, x: f32, y: f32) -> Option<f32> {
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

#[derive(Debug, Clone)]
pub struct LeNetWeights {
    pub conv1_weights: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weights: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub fc1_weights: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weights: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub fc3_weights: Vec<f32>,
    pub fc3_bias: Vec<f32>,
}

impl LeNetWeights {
    pub fn parameter_count(&self) -> usize {
        [
            &self.conv1_weights,
            &self.conv1_bias,
            &self.conv2_weights,
            &self.conv2_bias,
            &self.fc1_weights,
            &self.fc1_bias,
            &self.fc2_weights,
            &self.fc2_bias,
            &self.fc3_weights,
            &self.fc3_bias,
        ]
        .iter()
        .map(|v| v.len())
        .sum()
    }
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

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16777216.0
    }
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-6);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct LeNetClassifier {
    weights: LeNetWeights,
}

impl LeNetClassifier {
    pub fn new(weights: LeNetWeights) -> Result<Self> {
        weights.validate()?;
        Ok(Self { weights })
    }
    pub fn random(seed: u64) -> Self {
        Self {
            weights: LeNetWeights::random_init(seed),
        }
    }
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
    fn conv2d(
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
    fn max_pool(&self, input: &[f32], h: usize, w: usize, ch: usize) -> Vec<f32> {
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
    fn dense(&self, input: &[f32], weights: &[f32], bias: &[f32], relu: bool) -> Vec<f32> {
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
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - mx).exp()).sum();
        logits.iter().map(|&x| (x - mx).exp() / sum).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub predicted_class: usize,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

impl Prediction {
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = self
            .probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        v.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(k);
        v
    }
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    matrix: [[u32; NUM_CLASSES]; NUM_CLASSES],
    total: u32,
}
impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}
impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            matrix: [[0; NUM_CLASSES]; NUM_CLASSES],
            total: 0,
        }
    }
    pub fn record(&mut self, true_label: usize, predicted: usize) {
        if true_label < NUM_CLASSES && predicted < NUM_CLASSES {
            self.matrix[true_label][predicted] += 1;
            self.total += 1;
        }
    }
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (0..NUM_CLASSES).map(|i| self.matrix[i][i]).sum::<u32>() as f32 / self.total as f32
        }
    }
    pub fn precision(&self, c: usize) -> f32 {
        if c >= NUM_CLASSES {
            return 0.0;
        }
        let pp: u32 = (0..NUM_CLASSES).map(|i| self.matrix[i][c]).sum();
        if pp == 0 {
            0.0
        } else {
            self.matrix[c][c] as f32 / pp as f32
        }
    }
    pub fn recall(&self, c: usize) -> f32 {
        if c >= NUM_CLASSES {
            return 0.0;
        }
        let ap: u32 = self.matrix[c].iter().sum();
        if ap == 0 {
            0.0
        } else {
            self.matrix[c][c] as f32 / ap as f32
        }
    }
    pub fn f1_score(&self, c: usize) -> f32 {
        let (p, r) = (self.precision(c), self.recall(c));
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

fn stamp_pattern(pixels: &mut [f32], pattern: &[u8], pw: usize, ph: usize) {
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

fn main() {
    println!("=== Demo I: Handwriting Recognition (MNIST) ===\n");
    let classifier = LeNetClassifier::random(42);
    println!("Model parameters: {}", classifier.weights.parameter_count());
    for digit in 0..10u8 {
        let img = generate_test_digit(digit, 42 + u64::from(digit)).expect("gen");
        let pred = classifier
            .predict(&ImagePreprocessor::new().process(&img).expect("proc"))
            .expect("pred");
        println!(
            "Digit {digit}: predicted {} (conf: {:.1}%)",
            pred.predicted_class,
            pred.confidence * 100.0
        );
    }
    println!("\n=== Demo Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_image_creation() {
        assert!(GrayscaleImage::new(vec![0.5; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).is_ok());
    }
    #[test]
    fn test_image_wrong_size() {
        assert!(GrayscaleImage::new(vec![0.5; 100], IMAGE_WIDTH, IMAGE_HEIGHT).is_err());
    }
    #[test]
    fn test_image_nan() {
        let mut p = vec![0.5; IMAGE_SIZE];
        p[0] = f32::NAN;
        assert!(GrayscaleImage::new(p, IMAGE_WIDTH, IMAGE_HEIGHT).is_err());
    }
    #[test]
    fn test_mnist_bytes() {
        let img = GrayscaleImage::from_mnist_bytes(&vec![128u8; IMAGE_SIZE]).expect("ok");
        assert!((img.pixels[0] - 0.502).abs() < 0.01);
    }
    #[test]
    fn test_pixel_access() {
        let mut img =
            GrayscaleImage::new(vec![0.0; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).expect("ok");
        assert_eq!(img.get_pixel(0, 0), Some(0.0));
        assert_eq!(img.get_pixel(100, 100), None);
        img.set_pixel(5, 5, 0.75);
        assert_eq!(img.get_pixel(5, 5), Some(0.75));
    }
    #[test]
    fn test_preprocessor() {
        let img =
            GrayscaleImage::new(vec![0.3; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).expect("ok");
        let mut prep = ImagePreprocessor::new();
        prep.invert = true;
        prep.center = false;
        assert!((prep.process(&img).expect("ok").pixels[0] - 0.7).abs() < 0.01);
    }
    #[test]
    fn test_augmenter() {
        let img =
            GrayscaleImage::new(vec![0.0; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).expect("ok");
        let aug = ImageAugmenter::new(42);
        assert!(aug.rotate(&img, 45.0).is_ok());
        assert!(aug.scale(&img, 1.2).is_ok());
        let mut aug2 = ImageAugmenter::new(42);
        let noisy = aug2
            .add_noise(
                &GrayscaleImage::new(vec![0.5; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).expect("ok"),
                0.1,
            )
            .expect("ok");
        assert!(noisy
            .pixels
            .iter()
            .zip(std::iter::repeat(0.5_f32))
            .any(|(a, b)| (a - b).abs() > 0.001));
    }
    #[test]
    fn test_weights() {
        let w = LeNetWeights::random_init(42);
        assert!(w.validate().is_ok());
        assert_eq!(w.parameter_count(), 44426);
    }
    #[test]
    fn test_classifier_predict() {
        let c = LeNetClassifier::random(42);
        let pred = c
            .predict(&generate_test_digit(5, 42).expect("ok"))
            .expect("ok");
        assert!(pred.predicted_class < 10 && pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.probabilities.len(), 10);
        assert!((pred.probabilities.iter().sum::<f32>() - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_classifier_deterministic() {
        let c = LeNetClassifier::random(42);
        let img = generate_test_digit(3, 42).expect("ok");
        assert_eq!(
            c.predict(&img).expect("ok").predicted_class,
            c.predict(&img).expect("ok").predicted_class
        );
    }
    #[test]
    fn test_top_k() {
        let pred = LeNetClassifier::random(42)
            .predict(&generate_test_digit(7, 42).expect("ok"))
            .expect("ok");
        let top3 = pred.top_k(3);
        assert_eq!(top3.len(), 3);
        assert!(top3[0].1 >= top3[1].1);
    }
    #[test]
    fn test_confusion_matrix() {
        let mut cm = ConfusionMatrix::new();
        cm.record(0, 0);
        cm.record(0, 0);
        cm.record(0, 1);
        cm.record(1, 1);
        assert!((cm.accuracy() - 0.75).abs() < 0.01);
        assert!((cm.precision(0) - 1.0).abs() < 0.01);
        assert!((cm.recall(0) - 0.667).abs() < 0.01);
        assert!((cm.f1_score(0) - 0.8).abs() < 0.01);
    }
    #[test]
    fn test_generate_digits() {
        for d in 0..10 {
            assert!(generate_test_digit(d, 42).is_ok());
        }
        assert!(generate_test_digit(10, 42).is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test] fn prop_prediction_valid(seed in 0u64..1000) {
            let c = LeNetClassifier::random(seed);
            let pred = c.predict(&generate_test_digit((seed % 10) as u8, seed).expect("ok")).expect("ok");
            prop_assert!(pred.predicted_class < 10 && pred.probabilities.len() == 10);
            prop_assert!((pred.probabilities.iter().sum::<f32>() - 1.0).abs() < 0.01);
        }
        #[test] fn prop_rotation_preserves_dims(deg in -180.0f32..180.0f32) {
            let r = ImageAugmenter::new(42).rotate(&generate_test_digit(5, 42).expect("ok"), deg).expect("ok");
            prop_assert_eq!(r.pixels.len(), IMAGE_SIZE);
        }
        #[test] fn prop_noise_bounds(sigma in 0.01f32..0.5f32) {
            let noisy = ImageAugmenter::new(42).add_noise(&GrayscaleImage::new(vec![0.5; IMAGE_SIZE], IMAGE_WIDTH, IMAGE_HEIGHT).expect("ok"), sigma).expect("ok");
            for p in &noisy.pixels { prop_assert!(*p >= 0.0 && *p <= 1.0); }
        }
        #[test] fn prop_weights_valid(seed in 0u64..10000) { prop_assert!(LeNetWeights::random_init(seed).validate().is_ok()); }
    }
}
