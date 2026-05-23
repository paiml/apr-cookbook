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
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

// ============================================================================
// Style Transfer Engine
// ============================================================================

/// Style transfer engine
pub struct StyleTransfer {
    /// Gaussian blur kernel (read by main.rs tests for visibility check)
    pub blur_kernel: Kernel3x3,
    /// Edge detection kernel (Sobel X)
    pub sobel_x: Kernel3x3,
    /// Edge detection kernel (Sobel Y)
    pub sobel_y: Kernel3x3,
    /// Sharpen kernel
    pub sharpen_kernel: Kernel3x3,
}

impl StyleTransfer {
    /// Create new engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            blur_kernel: [
                [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
                [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
                [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            ],
            sobel_x: [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            sobel_y: [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            sharpen_kernel: [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        }
    }

    /// Apply style to image
    #[must_use]
    pub fn apply(&self, img: &Image, style: StylePreset, strength: f32) -> Image {
        let params = style.params();
        self.apply_with_params(img, &params, strength)
    }

    /// Apply style with custom parameters
    #[must_use]
    pub fn apply_with_params(&self, img: &Image, params: &StyleParams, strength: f32) -> Image {
        if img.is_empty() {
            return Image::new(0, 0);
        }

        let strength = strength.clamp(0.0, 1.0);

        // Step 1: Apply blur
        let mut result = if params.blur_amount > 0.5 {
            let iterations = (params.blur_amount / 1.0).ceil() as usize;
            let mut blurred = img.clone();
            for _ in 0..iterations {
                blurred = convolve_rgb(&blurred, &self.blur_kernel);
            }
            blurred
        } else {
            img.clone()
        };

        // Step 2: Edge enhancement
        if params.edge_strength > 0.0 {
            let edges = self.detect_edges(img);
            for (i, pixel) in result.pixels.iter_mut().enumerate() {
                let edge = edges[i];
                let edge_boost = edge * params.edge_strength;
                pixel.r = (pixel.r - edge_boost).clamp(0.0, 1.0);
                pixel.g = (pixel.g - edge_boost).clamp(0.0, 1.0);
                pixel.b = (pixel.b - edge_boost).clamp(0.0, 1.0);
            }
        }

        // Step 3: Color boost
        if (params.color_boost - 1.0).abs() > 0.01 {
            for pixel in &mut result.pixels {
                let lum = pixel.luminance();
                pixel.r = lum + (pixel.r - lum) * params.color_boost;
                pixel.g = lum + (pixel.g - lum) * params.color_boost;
                pixel.b = lum + (pixel.b - lum) * params.color_boost;
                pixel.r = pixel.r.clamp(0.0, 1.0);
                pixel.g = pixel.g.clamp(0.0, 1.0);
                pixel.b = pixel.b.clamp(0.0, 1.0);
            }
        }

        // Step 4: Posterization
        if params.posterize_levels < 255 {
            let levels = f32::from(params.posterize_levels);
            for pixel in &mut result.pixels {
                pixel.r = (pixel.r * levels).round() / levels;
                pixel.g = (pixel.g * levels).round() / levels;
                pixel.b = (pixel.b * levels).round() / levels;
            }
        }

        // Step 5: Blend with original based on strength
        if strength < 1.0 {
            for (i, pixel) in result.pixels.iter_mut().enumerate() {
                *pixel = img.pixels[i].blend(*pixel, strength);
            }
        }

        result
    }

    /// Detect edges using Sobel operator
    pub fn detect_edges(&self, img: &Image) -> Vec<f32> {
        // Convert to grayscale
        let gray: Vec<f32> = img.pixels.iter().map(|p| p.luminance()).collect();

        // Apply Sobel filters
        let gx = convolve_channel(&gray, img.width, img.height, &self.sobel_x);
        let gy = convolve_channel(&gray, img.width, img.height, &self.sobel_y);

        // Compute gradient magnitude
        gx.iter()
            .zip(gy.iter())
            .map(|(&x, &y)| (x * x + y * y).sqrt().min(1.0))
            .collect()
    }

    /// Apply sharpen filter
    #[must_use]
    pub fn sharpen(&self, img: &Image) -> Image {
        convolve_rgb(img, &self.sharpen_kernel)
    }

    /// Apply blur filter
    #[must_use]
    pub fn blur(&self, img: &Image) -> Image {
        convolve_rgb(img, &self.blur_kernel)
    }
}

impl Default for StyleTransfer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Style quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Color variety (0-1)
    pub color_variety: f32,
    /// Edge preservation (0-1)
    pub edge_preservation: f32,
    /// Style strength (0-1)
    pub style_strength: f32,
}

impl QualityMetrics {
    /// Evaluate styled image quality
    #[must_use]
    pub fn evaluate(original: &Image, styled: &Image) -> Self {
        if original.is_empty() || styled.is_empty() {
            return Self {
                color_variety: 0.0,
                edge_preservation: 0.0,
                style_strength: 0.0,
            };
        }

        // Color variety: standard deviation of colors
        let avg = styled.average_color();
        let variance: f32 = styled
            .pixels
            .iter()
            .map(|p| (p.r - avg.r).powi(2) + (p.g - avg.g).powi(2) + (p.b - avg.b).powi(2))
            .sum::<f32>()
            / styled.len() as f32;
        let color_variety = (variance.sqrt() * 2.0).min(1.0);

        // Edge preservation: correlation of luminance differences
        let mut edge_corr = 0.0_f32;
        let mut count = 0;
        for y in 1..original.height.min(styled.height) {
            for x in 1..original.width.min(styled.width) {
                if let (Some(orig), Some(orig_prev), Some(sty), Some(sty_prev)) = (
                    original.get(x, y),
                    original.get(x - 1, y),
                    styled.get(x, y),
                    styled.get(x - 1, y),
                ) {
                    let orig_diff = (orig.luminance() - orig_prev.luminance()).abs();
                    let sty_diff = (sty.luminance() - sty_prev.luminance()).abs();
                    if orig_diff > 0.01 {
                        edge_corr += 1.0 - (orig_diff - sty_diff).abs().min(1.0);
                        count += 1;
                    }
                }
            }
        }
        let edge_preservation = if count > 0 {
            edge_corr / count as f32
        } else {
            0.5
        };

        // Style strength: difference from original
        let mut total_diff = 0.0_f32;
        for (orig, sty) in original.pixels.iter().zip(styled.pixels.iter()) {
            total_diff += (orig.r - sty.r).abs() + (orig.g - sty.g).abs() + (orig.b - sty.b).abs();
        }
        let style_strength = (total_diff / (original.len() as f32 * 3.0) * 2.0).min(1.0);

        Self {
            color_variety,
            edge_preservation,
            style_strength,
        }
    }
}

// ============================================================================
// Image Generator (for testing)
// ============================================================================

/// Generate test images
pub struct ImageGenerator {
    pub rng: SimpleRng,
}

impl ImageGenerator {
    /// Create new generator
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SimpleRng::new(seed),
        }
    }

    /// Generate gradient image
    #[must_use]
    pub fn gradient(width: usize, height: usize) -> Image {
        let mut img = Image::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = x as f32 / width as f32;
                let g = y as f32 / height as f32;
                let b = 0.5;
                img.set(x, y, Pixel::new(r, g, b));
            }
        }
        img
    }

    /// Generate checkerboard pattern
    #[must_use]
    pub fn checkerboard(width: usize, height: usize, cell_size: usize) -> Image {
        let mut img = Image::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / cell_size) + (y / cell_size)) % 2 == 0;
                let color = if is_white { 1.0 } else { 0.0 };
                img.set(x, y, Pixel::new(color, color, color));
            }
        }
        img
    }

    /// Generate random noise image
    pub fn noise(&mut self, width: usize, height: usize) -> Image {
        let mut img = Image::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = self.rng.next_f32();
                let g = self.rng.next_f32();
                let b = self.rng.next_f32();
                img.set(x, y, Pixel::new(r, g, b));
            }
        }
        img
    }

    /// Generate circle image
    #[must_use]
    pub fn circle(width: usize, height: usize, radius: f32) -> Image {
        let mut img = Image::new(width, height);
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < radius {
                    img.set(x, y, Pixel::new(1.0, 0.0, 0.0));
                } else {
                    img.set(x, y, Pixel::new(0.2, 0.2, 0.8));
                }
            }
        }
        img
    }
}

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}
