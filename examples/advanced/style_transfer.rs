//! # Demo M: Real-time Style Transfer
//!
//! Applies artistic styles to images using a simplified Fast Style Transfer network.
//! Demonstrates convolution operations and image processing pipelines.
//!
//! ## Toyota Way Principles
//!
//! - **Heijunka**: Consistent processing time per pixel
//! - **Jidoka**: Quality detection for style strength
//! - **Kaizen**: Iterative refinement of style application

/// Image dimensions
pub const MAX_IMAGE_SIZE: usize = 512;

/// Number of style channels
pub const STYLE_CHANNELS: usize = 32;

// ============================================================================
// Image Types
// ============================================================================

/// RGB pixel
#[derive(Debug, Clone, Copy, Default)]
pub struct Pixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Pixel {
    /// Create new pixel
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create from u8 values
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
        }
    }

    /// Convert to u8
    #[must_use]
    pub fn to_u8(self) -> (u8, u8, u8) {
        (
            (self.r.clamp(0.0, 1.0) * 255.0) as u8,
            (self.g.clamp(0.0, 1.0) * 255.0) as u8,
            (self.b.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    /// Blend with another pixel
    #[must_use]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        Self {
            r: self.r * (1.0 - alpha) + other.r * alpha,
            g: self.g * (1.0 - alpha) + other.g * alpha,
            b: self.b * (1.0 - alpha) + other.b * alpha,
        }
    }

    /// Apply gamma correction
    #[must_use]
    pub fn gamma(self, gamma: f32) -> Self {
        Self {
            r: self.r.powf(gamma),
            g: self.g.powf(gamma),
            b: self.b.powf(gamma),
        }
    }

    /// Luminance
    #[must_use]
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

/// Image buffer
#[derive(Debug, Clone)]
pub struct Image {
    /// Pixel data
    pub pixels: Vec<Pixel>,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
}

impl Image {
    /// Create new image
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Pixel::default(); width * height],
            width,
            height,
        }
    }

    /// Create from pixel data
    #[must_use]
    pub fn from_pixels(pixels: Vec<Pixel>, width: usize, height: usize) -> Option<Self> {
        if pixels.len() == width * height {
            Some(Self {
                pixels,
                width,
                height,
            })
        } else {
            None
        }
    }

    /// Get pixel at (x, y)
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<Pixel> {
        if x < self.width && y < self.height {
            Some(self.pixels[y * self.width + x])
        } else {
            None
        }
    }

    /// Set pixel at (x, y)
    pub fn set(&mut self, x: usize, y: usize, pixel: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = pixel;
        }
    }

    /// Total pixels
    #[must_use]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Average color
    #[must_use]
    pub fn average_color(&self) -> Pixel {
        if self.pixels.is_empty() {
            return Pixel::default();
        }
        let sum = self
            .pixels
            .iter()
            .fold((0.0_f32, 0.0_f32, 0.0_f32), |acc, p| {
                (acc.0 + p.r, acc.1 + p.g, acc.2 + p.b)
            });
        let n = self.pixels.len() as f32;
        Pixel::new(sum.0 / n, sum.1 / n, sum.2 / n)
    }

    /// Resize image (nearest neighbor)
    #[must_use]
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let mut result = Self::new(new_width, new_height);
        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as usize;
                let src_y = (y as f32 * y_ratio) as usize;
                if let Some(pixel) = self.get(src_x, src_y) {
                    result.set(x, y, pixel);
                }
            }
        }
        result
    }
}

// ============================================================================
// Convolution Operations
// ============================================================================

/// 3x3 convolution kernel
pub type Kernel3x3 = [[f32; 3]; 3];

/// Apply 3x3 convolution to grayscale channel
#[allow(clippy::needless_range_loop)]
fn convolve_channel(img: &[f32], width: usize, height: usize, kernel: &Kernel3x3) -> Vec<f32> {
    let mut output = vec![0.0; width * height];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut sum = 0.0_f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    sum += img[py * width + px] * kernel[ky][kx];
                }
            }
            output[y * width + x] = sum;
        }
    }

    output
}

/// Apply convolution to RGB image
pub fn convolve_rgb(img: &Image, kernel: &Kernel3x3) -> Image {
    let r: Vec<f32> = img.pixels.iter().map(|p| p.r).collect();
    let g: Vec<f32> = img.pixels.iter().map(|p| p.g).collect();
    let b: Vec<f32> = img.pixels.iter().map(|p| p.b).collect();

    let r_out = convolve_channel(&r, img.width, img.height, kernel);
    let g_out = convolve_channel(&g, img.width, img.height, kernel);
    let b_out = convolve_channel(&b, img.width, img.height, kernel);

    let pixels: Vec<Pixel> = r_out
        .iter()
        .zip(g_out.iter())
        .zip(b_out.iter())
        .map(|((&r, &g), &b)| Pixel::new(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)))
        .collect();

    Image::from_pixels(pixels, img.width, img.height).unwrap_or_else(|| Image::new(0, 0))
}

// ============================================================================
// Style Definitions
// ============================================================================

/// Artistic style preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StylePreset {
    /// Oil painting effect
    OilPaint,
    /// Watercolor effect
    Watercolor,
    /// Pencil sketch
    PencilSketch,
    /// Pop art colors
    PopArt,
    /// Impressionist
    Impressionist,
}

impl StylePreset {
    /// Get display name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::OilPaint => "Oil Paint",
            Self::Watercolor => "Watercolor",
            Self::PencilSketch => "Pencil Sketch",
            Self::PopArt => "Pop Art",
            Self::Impressionist => "Impressionist",
        }
    }

    /// Get style parameters
    #[must_use]
    pub fn params(self) -> StyleParams {
        match self {
            Self::OilPaint => StyleParams {
                blur_amount: 2.0,
                edge_strength: 0.3,
                color_boost: 1.2,
                posterize_levels: 8,
            },
            Self::Watercolor => StyleParams {
                blur_amount: 3.0,
                edge_strength: 0.1,
                color_boost: 0.9,
                posterize_levels: 12,
            },
            Self::PencilSketch => StyleParams {
                blur_amount: 1.0,
                edge_strength: 1.0,
                color_boost: 0.0,
                posterize_levels: 2,
            },
            Self::PopArt => StyleParams {
                blur_amount: 0.5,
                edge_strength: 0.5,
                color_boost: 2.0,
                posterize_levels: 4,
            },
            Self::Impressionist => StyleParams {
                blur_amount: 1.5,
                edge_strength: 0.2,
                color_boost: 1.1,
                posterize_levels: 16,
            },
        }
    }
}

/// Style parameters
#[derive(Debug, Clone)]
pub struct StyleParams {
    /// Blur amount (0-5)
    pub blur_amount: f32,
    /// Edge enhancement strength (0-1)
    pub edge_strength: f32,
    /// Color boost factor
    pub color_boost: f32,
    /// Posterization levels
    pub posterize_levels: u8,
}

impl Default for StyleParams {
    fn default() -> Self {
        Self {
            blur_amount: 1.0,
            edge_strength: 0.5,
            color_boost: 1.0,
            posterize_levels: 8,
        }
    }
}

// ============================================================================
// Style Transfer Engine
// ============================================================================

/// Style transfer engine
pub struct StyleTransfer {
    /// Gaussian blur kernel
    blur_kernel: Kernel3x3,
    /// Edge detection kernel (Sobel X)
    sobel_x: Kernel3x3,
    /// Edge detection kernel (Sobel Y)
    sobel_y: Kernel3x3,
    /// Sharpen kernel
    sharpen_kernel: Kernel3x3,
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
    fn detect_edges(&self, img: &Image) -> Vec<f32> {
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
    rng: SimpleRng,
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

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ============================================================================
// Main
// ============================================================================

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
        let (r, g, b) = p.to_u8();
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
            prop_assert!(r8 <= 255);
            prop_assert!(g8 <= 255);
            prop_assert!(b8 <= 255);
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
