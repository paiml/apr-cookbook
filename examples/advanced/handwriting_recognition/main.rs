#![allow(unused_imports)]
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
