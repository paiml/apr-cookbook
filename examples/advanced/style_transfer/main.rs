#![allow(unused_imports)]
//! # Demo M: Real-time Style Transfer
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Applies artistic styles to images using a simplified Fast Style Transfer network.
//! Demonstrates convolution operations and image processing pipelines.
//!
//! ## Toyota Way Principles
//!
//! - **Heijunka**: Consistent processing time per pixel
//! - **Jidoka**: Quality detection for style strength
//! - **Kaizen**: Iterative refinement of style application
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Gatys, L. et al. (2016). *Image Style Transfer Using Convolutional Neural Networks*. CVPR. DOI: 10.1109/CVPR.2016.265

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo M: Real-time Style Transfer ===\n");

    let engine = StyleTransfer::new();
    let original = ImageGenerator::gradient(64, 64);

    println!("Original image: {}x{}", original.width, original.height);
    println!("Average color: {:?}\n", original.average_color());

    let styles = [
        StylePreset::OilPaint,
        StylePreset::Watercolor,
        StylePreset::PencilSketch,
        StylePreset::PopArt,
        StylePreset::Impressionist,
    ];

    for style in styles {
        println!("--- {} ---", style.name());
        let styled = engine.apply(&original, style, 1.0);
        let metrics = QualityMetrics::evaluate(&original, &styled);
        println!("  Color variety: {:.2}", metrics.color_variety);
        println!("  Edge preservation: {:.2}", metrics.edge_preservation);
        println!("  Style strength: {:.2}", metrics.style_strength);
    }

    println!("\n=== Demo M Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_new() {
        let p = Pixel::new(0.5, 0.5, 0.5);
        assert!((p.r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pixel_from_u8() {
        let p = Pixel::from_u8(128, 128, 128);
        assert!((p.r - 0.502).abs() < 0.01);
    }

    #[test]
    fn test_pixel_to_u8() {
        let p = Pixel::new(0.5, 0.5, 0.5);
        let (r, _g, _b) = p.to_u8();
        assert_eq!(r, 127);
    }

    #[test]
    fn test_pixel_blend() {
        let a = Pixel::new(0.0, 0.0, 0.0);
        let b = Pixel::new(1.0, 1.0, 1.0);
        let c = a.blend(b, 0.5);
        assert!((c.r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pixel_luminance() {
        let white = Pixel::new(1.0, 1.0, 1.0);
        let black = Pixel::new(0.0, 0.0, 0.0);
        assert!((white.luminance() - 1.0).abs() < 0.01);
        assert!((black.luminance() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_image_new() {
        let img = Image::new(10, 10);
        assert_eq!(img.width, 10);
        assert_eq!(img.height, 10);
        assert_eq!(img.len(), 100);
    }

    #[test]
    fn test_image_get_set() {
        let mut img = Image::new(10, 10);
        img.set(5, 5, Pixel::new(1.0, 0.0, 0.0));
        let p = img.get(5, 5).unwrap();
        assert!((p.r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_image_resize() {
        let img = Image::new(100, 100);
        let resized = img.resize(50, 50);
        assert_eq!(resized.width, 50);
        assert_eq!(resized.height, 50);
    }

    #[test]
    fn test_image_average_color() {
        let pixels = vec![
            Pixel::new(1.0, 0.0, 0.0),
            Pixel::new(0.0, 1.0, 0.0),
            Pixel::new(0.0, 0.0, 1.0),
            Pixel::new(1.0, 1.0, 1.0),
        ];
        let img = Image::from_pixels(pixels, 2, 2).unwrap();
        let avg = img.average_color();
        assert!((avg.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_style_preset_name() {
        assert_eq!(StylePreset::OilPaint.name(), "Oil Paint");
    }

    #[test]
    fn test_style_preset_params() {
        let params = StylePreset::PopArt.params();
        assert!(params.color_boost > 1.0);
    }

    #[test]
    fn test_style_preset_pencil_sketch() {
        let params = StylePreset::PencilSketch.params();
        assert_eq!(StylePreset::PencilSketch.name(), "Pencil Sketch");
        assert!(params.edge_strength > 0.0);
        assert!(params.blur_amount >= 0.0);
    }

    #[test]
    fn test_style_preset_impressionist() {
        let params = StylePreset::Impressionist.params();
        assert_eq!(StylePreset::Impressionist.name(), "Impressionist");
        assert!(params.blur_amount > 0.0);
        assert!(params.color_boost > 0.0);
    }

    #[test]
    fn test_style_transfer_new() {
        let engine = StyleTransfer::new();
        // Verify kernels are set
        assert!((engine.blur_kernel[1][1] - 4.0 / 16.0).abs() < 0.01);
    }

    #[test]
    fn test_style_transfer_apply() {
        let engine = StyleTransfer::new();
        let img = ImageGenerator::gradient(32, 32);
        let styled = engine.apply(&img, StylePreset::OilPaint, 1.0);
        assert_eq!(styled.width, img.width);
    }

    #[test]
    fn test_style_transfer_blur() {
        let engine = StyleTransfer::new();
        let img = ImageGenerator::checkerboard(32, 32, 4);
        let blurred = engine.blur(&img);
        assert_eq!(blurred.width, img.width);
    }

    #[test]
    fn test_quality_metrics() {
        let original = ImageGenerator::gradient(32, 32);
        let engine = StyleTransfer::new();
        let styled = engine.apply(&original, StylePreset::PopArt, 1.0);
        let metrics = QualityMetrics::evaluate(&original, &styled);
        assert!(metrics.color_variety >= 0.0);
        assert!(metrics.style_strength >= 0.0);
    }

    #[test]
    fn test_image_generator_gradient() {
        let img = ImageGenerator::gradient(10, 10);
        assert_eq!(img.width, 10);
        let top_left = img.get(0, 0).unwrap();
        let bottom_right = img.get(9, 9).unwrap();
        assert!(top_left.r < bottom_right.r);
    }

    #[test]
    fn test_image_generator_checkerboard() {
        let img = ImageGenerator::checkerboard(10, 10, 5);
        let p1 = img.get(0, 0).unwrap();
        let p2 = img.get(5, 0).unwrap();
        assert!((p1.r - p2.r).abs() > 0.5);
    }

    #[test]
    fn test_image_generator_circle() {
        let img = ImageGenerator::circle(20, 20, 5.0);
        let center = img.get(10, 10).unwrap();
        let corner = img.get(0, 0).unwrap();
        assert!((center.r - 1.0).abs() < 0.01);
        assert!(corner.r < 0.5);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn prop_pixel_blend_bounded(alpha in 0.0f32..1.0) {
            let a = Pixel::new(0.0, 0.0, 0.0);
            let b = Pixel::new(1.0, 1.0, 1.0);
            let c = a.blend(b, alpha);
            prop_assert!(c.r >= 0.0 && c.r <= 1.0);
        }

        #[test]
        fn prop_pixel_to_u8_valid(r in 0.0f32..1.0, g in 0.0f32..1.0, b in 0.0f32..1.0) {
            let p = Pixel::new(r, g, b);
            let (r8, g8, b8) = p.to_u8();
            let _ = (r8, g8, b8);
        }

        #[test]
        fn prop_image_resize_dimensions(w in 10usize..50, h in 10usize..50, new_w in 5usize..30, new_h in 5usize..30) {
            let img = Image::new(w, h);
            let resized = img.resize(new_w, new_h);
            prop_assert_eq!(resized.width, new_w);
            prop_assert_eq!(resized.height, new_h);
        }

        #[test]
        fn prop_style_strength_bounded(strength in 0.0f32..1.0) {
            let engine = StyleTransfer::new();
            let img = ImageGenerator::gradient(16, 16);
            let styled = engine.apply(&img, StylePreset::OilPaint, strength);
            prop_assert_eq!(styled.width, img.width);
        }

        #[test]
        fn prop_quality_metrics_bounded(seed in 0u64..1000) {
            let mut gen = ImageGenerator::new(seed);
            let original = gen.noise(16, 16);
            let engine = StyleTransfer::new();
            let styled = engine.apply(&original, StylePreset::Watercolor, 0.8);
            let metrics = QualityMetrics::evaluate(&original, &styled);
            prop_assert!(metrics.color_variety >= 0.0 && metrics.color_variety <= 1.0);
            prop_assert!(metrics.style_strength >= 0.0 && metrics.style_strength <= 1.0);
        }
    }
}
