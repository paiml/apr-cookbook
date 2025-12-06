//! Format conversion utilities for ML models.
//!
//! This module provides bidirectional conversion between:
//! - `SafeTensors` (Hugging Face standard)
//! - APR (native format)
//! - GGUF (llama.cpp ecosystem)
//!
//! # Philosophy (Genchi Genbutsu)
//!
//! By supporting multiple formats, we meet models where they are:
//! - Hugging Face Hub → `SafeTensors`
//! - Local training → APR
//! - llama.cpp deployment → GGUF

use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

/// Supported conversion formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionFormat {
    /// `SafeTensors` format (Hugging Face)
    SafeTensors,
    /// APR format (native)
    Apr,
    /// GGUF format (llama.cpp)
    Gguf,
}

impl ConversionFormat {
    /// Get the file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Apr => "apr",
            Self::Gguf => "gguf",
        }
    }

    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "safetensors" => Some(Self::SafeTensors),
            "apr" => Some(Self::Apr),
            "gguf" => Some(Self::Gguf),
            _ => None,
        }
    }

    /// Detect format from path.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
    }
}

impl std::fmt::Display for ConversionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::Apr => write!(f, "APR"),
            Self::Gguf => write!(f, "GGUF"),
        }
    }
}

/// Tensor data for conversion operations.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Raw bytes
    pub data: Vec<u8>,
}

/// Supported data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// 16-bit brain float
    BF16,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// Quantized 8-bit (GGUF `Q8_0`)
    Q8_0,
    /// Quantized 4-bit (GGUF `Q4_0`)
    Q4_0,
}

impl DataType {
    /// Get bytes per element for this dtype.
    ///
    /// Note: Quantized types have variable size per block,
    /// so these values are approximate.
    #[must_use]
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 | Self::Q8_0 | Self::Q4_0 => 1,
        }
    }
}

/// Model converter for format transformations.
#[derive(Debug)]
pub struct AprConverter {
    /// Named tensors
    tensors: HashMap<String, TensorData>,
    /// Model metadata
    metadata: ConversionMetadata,
}

/// Metadata for conversion operations.
#[derive(Debug, Clone, Default)]
pub struct ConversionMetadata {
    /// Model name
    pub name: Option<String>,
    /// Model architecture
    pub architecture: Option<String>,
    /// Source format
    pub source_format: Option<ConversionFormat>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl AprConverter {
    /// Create a new converter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: ConversionMetadata::default(),
        }
    }

    /// Add a tensor to the converter.
    pub fn add_tensor(&mut self, tensor: TensorData) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, metadata: ConversionMetadata) {
        self.metadata = metadata;
    }

    /// Get all tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Get a tensor by name.
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Get the number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Calculate total parameter count.
    #[must_use]
    pub fn total_parameters(&self) -> usize {
        self.tensors
            .values()
            .map(|t| t.shape.iter().product::<usize>())
            .sum()
    }

    /// Convert to APR format bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails (currently infallible).
    pub fn to_apr(&self) -> Result<Vec<u8>> {
        use crate::bundle::ModelBundle;

        // For now, create a simple APR bundle
        // In production, this would properly serialize all tensors
        let payload = self.serialize_tensors();

        let mut bundle = ModelBundle::new().with_payload(payload);

        if let Some(name) = &self.metadata.name {
            bundle = bundle.with_name(name);
        }

        Ok(bundle.build())
    }

    /// Serialize tensors to a simple binary format.
    fn serialize_tensors(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write tensor count
        let count = self.tensors.len() as u32;
        bytes.extend_from_slice(&count.to_le_bytes());

        // Write each tensor
        for (name, tensor) in &self.tensors {
            // Name length and bytes
            let name_bytes = name.as_bytes();
            bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(name_bytes);

            // Shape
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for &dim in &tensor.shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // Data
            bytes.extend_from_slice(&(tensor.data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&tensor.data);
        }

        bytes
    }

    /// Check if a conversion path is supported.
    #[must_use]
    pub fn is_conversion_supported(from: ConversionFormat, to: ConversionFormat) -> bool {
        matches!(
            (from, to),
            (
                ConversionFormat::SafeTensors | ConversionFormat::Gguf,
                ConversionFormat::Apr
            ) | (
                ConversionFormat::Apr,
                ConversionFormat::Gguf | ConversionFormat::SafeTensors
            )
        )
    }
}

impl Default for AprConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_conversion_format_extension() {
        assert_eq!(ConversionFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ConversionFormat::Apr.extension(), "apr");
        assert_eq!(ConversionFormat::Gguf.extension(), "gguf");
    }

    #[test]
    fn test_conversion_format_from_extension() {
        assert_eq!(
            ConversionFormat::from_extension("safetensors"),
            Some(ConversionFormat::SafeTensors)
        );
        assert_eq!(
            ConversionFormat::from_extension("apr"),
            Some(ConversionFormat::Apr)
        );
        assert_eq!(
            ConversionFormat::from_extension("gguf"),
            Some(ConversionFormat::Gguf)
        );
        assert_eq!(ConversionFormat::from_extension("txt"), None);
    }

    #[test]
    fn test_conversion_format_from_extension_case_insensitive() {
        assert_eq!(
            ConversionFormat::from_extension("SAFETENSORS"),
            Some(ConversionFormat::SafeTensors)
        );
        assert_eq!(
            ConversionFormat::from_extension("APR"),
            Some(ConversionFormat::Apr)
        );
    }

    #[test]
    fn test_conversion_format_from_path() {
        let path = PathBuf::from("/models/classifier.safetensors");
        assert_eq!(
            ConversionFormat::from_path(&path),
            Some(ConversionFormat::SafeTensors)
        );

        let path = PathBuf::from("/models/model.apr");
        assert_eq!(
            ConversionFormat::from_path(&path),
            Some(ConversionFormat::Apr)
        );

        let path = PathBuf::from("/models/llama.gguf");
        assert_eq!(
            ConversionFormat::from_path(&path),
            Some(ConversionFormat::Gguf)
        );
    }

    #[test]
    fn test_conversion_format_display() {
        assert_eq!(format!("{}", ConversionFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", ConversionFormat::Apr), "APR");
        assert_eq!(format!("{}", ConversionFormat::Gguf), "GGUF");
    }

    #[test]
    fn test_data_type_element_size() {
        assert_eq!(DataType::F32.element_size(), 4);
        assert_eq!(DataType::F16.element_size(), 2);
        assert_eq!(DataType::BF16.element_size(), 2);
        assert_eq!(DataType::I8.element_size(), 1);
        assert_eq!(DataType::U8.element_size(), 1);
    }

    #[test]
    fn test_apr_converter_add_tensor() {
        let mut converter = AprConverter::new();

        let tensor = TensorData {
            name: "weights".to_string(),
            shape: vec![784, 128],
            dtype: DataType::F32,
            data: vec![0u8; 784 * 128 * 4],
        };

        converter.add_tensor(tensor);
        assert_eq!(converter.tensor_count(), 1);
        assert!(converter.get_tensor("weights").is_some());
    }

    #[test]
    fn test_apr_converter_tensor_names() {
        let mut converter = AprConverter::new();

        converter.add_tensor(TensorData {
            name: "layer1.weight".to_string(),
            shape: vec![128, 64],
            dtype: DataType::F32,
            data: vec![],
        });

        converter.add_tensor(TensorData {
            name: "layer1.bias".to_string(),
            shape: vec![64],
            dtype: DataType::F32,
            data: vec![],
        });

        let names = converter.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"layer1.weight"));
        assert!(names.contains(&"layer1.bias"));
    }

    #[test]
    fn test_apr_converter_total_parameters() {
        let mut converter = AprConverter::new();

        converter.add_tensor(TensorData {
            name: "weights".to_string(),
            shape: vec![100, 50],
            dtype: DataType::F32,
            data: vec![],
        });

        converter.add_tensor(TensorData {
            name: "bias".to_string(),
            shape: vec![50],
            dtype: DataType::F32,
            data: vec![],
        });

        assert_eq!(converter.total_parameters(), 100 * 50 + 50);
    }

    #[test]
    fn test_apr_converter_to_apr() {
        let mut converter = AprConverter::new();
        converter.set_metadata(ConversionMetadata {
            name: Some("test-model".to_string()),
            ..Default::default()
        });

        converter.add_tensor(TensorData {
            name: "weights".to_string(),
            shape: vec![10, 5],
            dtype: DataType::F32,
            data: vec![0u8; 200],
        });

        let apr_bytes = converter.to_apr().unwrap();

        // Verify it starts with APR magic
        assert_eq!(&apr_bytes[0..4], b"APRN");
    }

    #[test]
    fn test_is_conversion_supported() {
        // Supported conversions
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::SafeTensors,
            ConversionFormat::Apr
        ));
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Apr,
            ConversionFormat::Gguf
        ));
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Gguf,
            ConversionFormat::Apr
        ));
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Apr,
            ConversionFormat::SafeTensors
        ));

        // Unsupported direct conversions
        assert!(!AprConverter::is_conversion_supported(
            ConversionFormat::SafeTensors,
            ConversionFormat::Gguf
        ));
        assert!(!AprConverter::is_conversion_supported(
            ConversionFormat::Gguf,
            ConversionFormat::SafeTensors
        ));
    }

    #[test]
    fn test_conversion_metadata_default() {
        let metadata = ConversionMetadata::default();
        assert!(metadata.name.is_none());
        assert!(metadata.architecture.is_none());
        assert!(metadata.source_format.is_none());
        assert!(metadata.custom.is_empty());
    }

    #[test]
    fn test_tensor_data_creation() {
        let tensor = TensorData {
            name: "test".to_string(),
            shape: vec![2, 3, 4],
            dtype: DataType::F32,
            data: vec![0u8; 2 * 3 * 4 * 4],
        };

        assert_eq!(tensor.name, "test");
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.dtype, DataType::F32);
        assert_eq!(tensor.data.len(), 96); // 2*3*4*4 bytes
    }
}
