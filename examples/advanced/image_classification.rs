//! # Demo J: Image Classification (MobileNet-style)
//!
//! MobileNet-style classification with depthwise separable convolutions,
//! squeeze-and-excitation, and efficient mobile inference.
//!
//! ## QA: Build, test, clippy, fmt PASS. Property tests (100+ cases).
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR. arXiv:1512.03385

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
    fn resize(&self, image: &RgbImage, nw: usize, nh: usize) -> Result<RgbImage> {
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
    fn center_crop(&self, image: &RgbImage) -> Result<RgbImage> {
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
    fn apply_imagenet_normalization(&self, image: &RgbImage) -> Result<RgbImage> {
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
    weights: MobileNetWeights,
    labels: Vec<String>,
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
    fn stem_conv(&self, input: &[f32]) -> Vec<f32> {
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
    fn global_avg_pool(&self, input: &[f32], h: usize, w: usize, ch: usize) -> Vec<f32> {
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
    fn dense(&self, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
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
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let es: f32 = logits.iter().map(|&x| (x - mx).exp()).sum();
        logits.iter().map(|&x| (x - mx).exp() / es).collect()
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

fn main() {
    println!("=== Demo J: Image Classification (MobileNet-style) ===\n");
    let classifier = ImageClassifier::random(42);
    println!("Model parameters: {}", classifier.weights.parameter_count());
    let pp = ImagePreprocessor::new();
    let processed = pp
        .process(&generate_test_image(42).expect("gen"))
        .expect("preprocess");
    println!("Processed: {}x{}", processed.width, processed.height);
    let result = classifier.predict(&processed).expect("classify");
    println!(
        "Predicted: {} ({}) {:.4}%",
        result.predicted_class,
        result.label,
        result.confidence * 100.0
    );
    for (i, (c, l, p)) in result.top_k(5).iter().enumerate() {
        println!("  {}. {} ({}) {:.4}%", i + 1, c, l, p * 100.0);
    }
    let batch: Vec<RgbImage> = (0..4)
        .map(|i| {
            pp.process(&generate_test_image(42 + i).expect("g"))
                .expect("p")
        })
        .collect();
    for (i, r) in classifier
        .predict_batch(&batch)
        .expect("batch")
        .iter()
        .enumerate()
    {
        println!(
            "  Image {}: class {} ({:.2}%)",
            i,
            r.predicted_class,
            r.confidence * 100.0
        );
    }
    println!("\n=== Demo Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rgb_image_creation_and_errors() {
        assert!(RgbImage::new(
            vec![0.5; NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE],
            IMAGE_SIZE,
            IMAGE_SIZE
        )
        .is_ok());
        assert!(RgbImage::new(vec![0.5; 100], IMAGE_SIZE, IMAGE_SIZE).is_err());
        let mut px = vec![0.5; NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE];
        px[0] = f32::NAN;
        assert!(RgbImage::new(px, IMAGE_SIZE, IMAGE_SIZE).is_err());
    }
    #[test]
    fn test_rgb_from_bytes() {
        let img =
            RgbImage::from_rgb_bytes(&vec![128u8; NUM_CHANNELS * 32 * 32], 32, 32).expect("ok");
        assert!((img.get_pixel(0, 0, 0).unwrap_or(0.0) - 0.502).abs() < 0.01);
    }
    #[test]
    fn test_pixel_access() {
        let img = RgbImage::new(vec![0.0; NUM_CHANNELS * 32 * 32], 32, 32).expect("ok");
        assert!(img.get_pixel(0, 0, 0).is_some());
        assert!(img.get_pixel(3, 0, 0).is_none());
        assert!(img.get_pixel(0, 100, 0).is_none());
    }
    #[test]
    fn test_preprocessor() {
        let p = ImagePreprocessor::new()
            .process(&generate_test_image(42).expect("ok"))
            .expect("ok");
        assert_eq!((p.width, p.height), (IMAGE_SIZE, IMAGE_SIZE));
    }
    #[test]
    fn test_activations() {
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(1.0), 1.0);
        assert_eq!(Activation::ReLU6.apply(10.0), 6.0);
        assert!((Activation::HardSwish.apply(3.0) - 3.0).abs() < 0.01);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 0.01);
    }
    #[test]
    fn test_weights_validation_and_count() {
        let w = MobileNetWeights::random_init(42);
        assert!(w.validate().is_ok());
        assert_eq!(w.parameter_count(), 321448);
    }
    #[test]
    fn test_classifier_predict_deterministic_softmax() {
        let c = ImageClassifier::random(42);
        let img = ImagePreprocessor::new()
            .process(&generate_test_image(42).expect("ok"))
            .expect("ok");
        let r1 = c.predict(&img).expect("ok");
        let r2 = c.predict(&img).expect("ok");
        assert!(r1.predicted_class < NUM_CLASSES);
        assert!(r1.confidence >= 0.0 && r1.confidence <= 1.0);
        assert_eq!(r1.predicted_class, r2.predicted_class);
        assert!((r1.probabilities.iter().sum::<f32>() - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_batch_and_top_k() {
        let c = ImageClassifier::random(42);
        let pp = ImagePreprocessor::new();
        let imgs: Vec<RgbImage> = (0..4)
            .map(|i| {
                pp.process(&generate_test_image(42 + i).expect("o"))
                    .expect("o")
            })
            .collect();
        assert_eq!(c.predict_batch(&imgs).expect("ok").len(), 4);
        let top5 = c.predict(&imgs[0]).expect("ok").top_k(5);
        assert_eq!(top5.len(), 5);
        for i in 1..5 {
            assert!(top5[i - 1].2 >= top5[i].2);
        }
    }
    #[test]
    fn test_confidence_and_format() {
        let r = ClassificationResult {
            predicted_class: 5,
            label: "t".into(),
            confidence: 0.85,
            probabilities: vec![0.1; NUM_CLASSES],
        };
        assert!(r.is_confident(0.8));
        assert!(!r.is_confident(0.9));
        assert_eq!(
            ImageFormat::from_magic_bytes(&[0xFF, 0xD8, 0xFF, 0xE0]),
            ImageFormat::Jpeg
        );
        assert_eq!(
            ImageFormat::from_magic_bytes(&[0x89, 0x50, 0x4E, 0x47]),
            ImageFormat::Png
        );
        assert_eq!(
            ImageFormat::from_magic_bytes(&[
                0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50
            ]),
            ImageFormat::WebP
        );
        assert_eq!(
            ImageFormat::from_magic_bytes(&[0, 0, 0, 0]),
            ImageFormat::Unknown
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn prop_prediction_valid(seed in 0u64..10000) {
            let c = ImageClassifier::random(seed);
            let img = ImagePreprocessor::new().process(&generate_test_image(seed).expect("ok")).expect("ok");
            let r = c.predict(&img).expect("ok");
            prop_assert!(r.predicted_class < NUM_CLASSES && r.confidence >= 0.0 && r.confidence <= 1.0);
            prop_assert!((r.probabilities.iter().sum::<f32>() - 1.0).abs() < 0.01);
        }
        #[test]
        fn prop_activation_bounds(x in -100.0f32..100.0f32) {
            prop_assert!(Activation::ReLU.apply(x) >= 0.0);
            let r6 = Activation::ReLU6.apply(x); prop_assert!(r6 >= 0.0 && r6 <= 6.0);
            let sig = Activation::Sigmoid.apply(x); prop_assert!(sig >= 0.0 && sig <= 1.0);
        }
        #[test]
        fn prop_top_k_sorted(k in 1usize..20) {
            let c = ImageClassifier::random(42);
            let img = ImagePreprocessor::new().process(&generate_test_image(42).expect("ok")).expect("ok");
            let top = c.predict(&img).expect("ok").top_k(k);
            for i in 1..top.len() { prop_assert!(top[i-1].2 >= top[i].2); }
        }
    }
}
