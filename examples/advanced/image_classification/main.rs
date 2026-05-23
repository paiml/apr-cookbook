#![allow(unused_imports)]
//! # Demo J: Image Classification (MobileNet-style)
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
