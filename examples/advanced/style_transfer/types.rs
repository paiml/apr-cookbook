#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
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
pub fn convolve_channel(img: &[f32], width: usize, height: usize, kernel: &Kernel3x3) -> Vec<f32> {
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
