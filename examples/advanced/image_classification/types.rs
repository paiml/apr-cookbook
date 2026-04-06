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

pub const IMAGE_SIZE: usize = 224;
pub const NUM_CHANNELS: usize = 3;
pub const NUM_CLASSES: usize = 1000;
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationError {
    InvalidDimensions {
        expected_width: usize,
        expected_height: usize,
        got_width: usize,
        got_height: usize,
    },
    InvalidChannels {
        expected: usize,
        got: usize,
    },
    InvalidPixelValue {
        channel: usize,
        index: usize,
        value: f32,
    },
    WeightError(String),
    PreprocessingError(String),
    InferenceError(String),
}
impl std::fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions {
                expected_width,
                expected_height,
                got_width,
                got_height,
            } => write!(
                f,
                "Invalid dimensions: expected {}x{}, got {}x{}",
                expected_width, expected_height, got_width, got_height
            ),
            Self::InvalidChannels { expected, got } => {
                write!(f, "Invalid channels: expected {expected}, got {got}")
            }
            Self::InvalidPixelValue {
                channel,
                index,
                value,
            } => write!(f, "Invalid pixel at ch {channel}, idx {index}: {value}"),
            Self::WeightError(m) => write!(f, "Weight error: {m}"),
            Self::PreprocessingError(m) => write!(f, "Preprocessing error: {m}"),
            Self::InferenceError(m) => write!(f, "Inference error: {m}"),
        }
    }
}
impl std::error::Error for ClassificationError {}
pub type Result<T> = std::result::Result<T, ClassificationError>;

#[derive(Debug, Clone)]
pub struct RgbImage {
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}
impl RgbImage {
    pub fn new(pixels: Vec<f32>, width: usize, height: usize) -> Result<Self> {
        let expected = NUM_CHANNELS * height * width;
        if pixels.len() != expected {
            return Err(ClassificationError::InvalidDimensions {
                expected_width: width,
                expected_height: height,
                got_width: pixels.len() / (height * NUM_CHANNELS),
                got_height: height,
            });
        }
        for c in 0..NUM_CHANNELS {
            for i in 0..(height * width) {
                let p = pixels[c * height * width + i];
                if p.is_nan() || p.is_infinite() {
                    return Err(ClassificationError::InvalidPixelValue {
                        channel: c,
                        index: i,
                        value: p,
                    });
                }
            }
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels: NUM_CHANNELS,
        })
    }
    pub fn from_rgb_bytes(bytes: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = NUM_CHANNELS * height * width;
        if bytes.len() != expected {
            return Err(ClassificationError::InvalidDimensions {
                expected_width: width,
                expected_height: height,
                got_width: bytes.len() / (height * NUM_CHANNELS),
                got_height: height,
            });
        }
        let mut pixels = vec![0.0_f32; expected];
        for c in 0..NUM_CHANNELS {
            for y in 0..height {
                for x in 0..width {
                    pixels[c * height * width + y * width + x] =
                        f32::from(bytes[(y * width + x) * NUM_CHANNELS + c]) / 255.0;
                }
            }
        }
        Self::new(pixels, width, height)
    }
    pub fn get_pixel(&self, channel: usize, y: usize, x: usize) -> Option<f32> {
        if channel < self.channels && y < self.height && x < self.width {
            Some(self.pixels[channel * self.height * self.width + y * self.width + x])
        } else {
            None
        }
    }
    pub fn len(&self) -> usize {
        self.pixels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    pub target_width: usize,
    pub target_height: usize,
    pub normalize: bool,
    pub crop_ratio: f32,
}
impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self::new()
    }
}
impl ImagePreprocessor {
    pub fn new() -> Self {
        Self {
            target_width: IMAGE_SIZE,
            target_height: IMAGE_SIZE,
            normalize: true,
            crop_ratio: 256.0 / 224.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    ReLU6,
    HardSwish,
    Sigmoid,
    None,
}
impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::ReLU6 => x.clamp(0.0, 6.0),
            Self::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::None => x,
        }
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
        let (u1, u2) = (self.next_f32().max(1e-6), self.next_f32());
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct MobileNetWeights {
    pub stem_weights: Vec<f32>,
    pub stem_bias: Vec<f32>,
    pub classifier_weights: Vec<f32>,
    pub classifier_bias: Vec<f32>,
    pub feature_dim: usize,
}
impl MobileNetWeights {
    pub fn random_init(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let sw: Vec<f32> = (0..432)
            .map(|_| rng.next_gaussian() * (2.0 / 27.0_f32).sqrt())
            .collect();
        let cw: Vec<f32> = (0..320000)
            .map(|_| rng.next_gaussian() * (2.0 / 320.0_f32).sqrt())
            .collect();
        Self {
            stem_weights: sw,
            stem_bias: vec![0.0; 16],
            classifier_weights: cw,
            classifier_bias: vec![0.0; NUM_CLASSES],
            feature_dim: 320,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.stem_weights.len() != 432 {
            return Err(ClassificationError::WeightError(format!(
                "stem: expected 432, got {}",
                self.stem_weights.len()
            )));
        }
        if self.classifier_weights.len() != self.feature_dim * NUM_CLASSES {
            return Err(ClassificationError::WeightError(format!(
                "classifier: expected {}, got {}",
                self.feature_dim * NUM_CLASSES,
                self.classifier_weights.len()
            )));
        }
        let chk = |s: &[f32], n: &str| -> Result<()> {
            for (i, &w) in s.iter().enumerate() {
                if w.is_nan() || w.is_infinite() {
                    return Err(ClassificationError::WeightError(format!(
                        "Invalid {n} at {i}: {w}"
                    )));
                }
            }
            Ok(())
        };
        chk(&self.stem_weights, "stem")?;
        chk(&self.classifier_weights, "classifier")
    }
    pub fn parameter_count(&self) -> usize {
        self.stem_weights.len()
            + self.stem_bias.len()
            + self.classifier_weights.len()
            + self.classifier_bias.len()
    }
}

#[derive(Debug, Clone)]
pub struct ImageClassifier {
    pub weights: MobileNetWeights,
    pub labels: Vec<String>,
}
impl ImageClassifier {
    pub fn new(weights: MobileNetWeights, labels: Vec<String>) -> Result<Self> {
        weights.validate()?;
        if labels.len() != NUM_CLASSES {
            return Err(ClassificationError::InferenceError(format!(
                "Expected {} labels, got {}",
                NUM_CLASSES,
                labels.len()
            )));
        }
        Ok(Self { weights, labels })
    }
    pub fn random(seed: u64) -> Self {
        Self {
            weights: MobileNetWeights::random_init(seed),
            labels: (0..NUM_CLASSES).map(|i| format!("class_{i}")).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub predicted_class: usize,
    pub label: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}
impl ClassificationResult {
    pub fn top_k(&self, k: usize) -> Vec<(usize, String, f32)> {
        let mut idx: Vec<(usize, f32)> = self
            .probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        idx.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(k);
        idx.into_iter()
            .map(|(i, p)| (i, format!("class_{i}"), p))
            .collect()
    }
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Unknown,
}
impl ImageFormat {
    pub fn from_magic_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < 4 {
            return Self::Unknown;
        }
        if bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
            return Self::Jpeg;
        }
        if bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47 {
            return Self::Png;
        }
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            return Self::WebP;
        }
        Self::Unknown
    }
}
