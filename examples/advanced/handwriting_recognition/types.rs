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
use proptest::prelude::*;
#[allow(unused_imports)]
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
}

#[derive(Debug, Clone)]
pub struct ImageAugmenter {
    pub seed: u64,
}

impl ImageAugmenter {
    pub fn new(seed: u64) -> Self {
        Self { seed }
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
}

pub struct SimpleRng {
    pub state: u64,
}
impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16777216.0
    }
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-6);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct LeNetClassifier {
    pub weights: LeNetWeights,
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
    pub total: u32,
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
