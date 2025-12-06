//! Property-based tests for the convert module.
//!
//! These tests verify conversion invariants.

#![allow(clippy::disallowed_methods)]

use apr_cookbook::convert::{AprConverter, ConversionFormat, DataType, TensorData};
use proptest::prelude::*;

/// Strategy for generating valid tensor names
fn tensor_name() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_.]{0,127}".prop_map(|s| s.clone())
}

/// Strategy for generating valid tensor shapes (1-4 dimensions, reasonable sizes)
fn tensor_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1..64usize, 1..4)
}

/// Strategy for generating data types
fn data_type() -> impl Strategy<Value = DataType> {
    prop_oneof![
        Just(DataType::F32),
        Just(DataType::F16),
        Just(DataType::BF16),
        Just(DataType::I8),
        Just(DataType::U8),
    ]
}

/// Strategy for generating a single tensor
fn tensor_data() -> impl Strategy<Value = TensorData> {
    (tensor_name(), tensor_shape(), data_type()).prop_map(|(name, shape, dtype)| {
        let num_elements: usize = shape.iter().product();
        let elem_size = dtype.element_size();
        let data = vec![0u8; num_elements * elem_size];
        TensorData {
            name,
            shape,
            dtype,
            data,
        }
    })
}

proptest! {
    /// Property: Adding tensors increases tensor count
    #[test]
    fn adding_tensors_increases_count(tensors in prop::collection::vec(tensor_data(), 1..5)) {
        let mut converter = AprConverter::new();

        for (i, tensor) in tensors.iter().enumerate() {
            converter.add_tensor(tensor.clone());
            prop_assert_eq!(converter.tensor_count(), i + 1);
        }
    }

    /// Property: Total parameters equals sum of tensor elements
    #[test]
    fn total_params_equals_sum(tensors in prop::collection::vec(tensor_data(), 1..5)) {
        let mut converter = AprConverter::new();

        let expected_params: usize = tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();

        for tensor in tensors {
            converter.add_tensor(tensor);
        }

        prop_assert_eq!(converter.total_parameters(), expected_params);
    }

    /// Property: Tensor names are preserved
    #[test]
    fn tensor_names_preserved(tensors in prop::collection::vec(tensor_data(), 1..5)) {
        let mut converter = AprConverter::new();
        let expected_names: Vec<String> = tensors.iter().map(|t| t.name.clone()).collect();

        for tensor in tensors {
            converter.add_tensor(tensor);
        }

        let actual_names = converter.tensor_names();
        prop_assert_eq!(actual_names.len(), expected_names.len());

        for name in expected_names {
            prop_assert!(actual_names.contains(&name.as_str()));
        }
    }

    /// Property: Conversion to APR produces valid header
    #[test]
    fn conversion_produces_valid_apr(tensor in tensor_data()) {
        let mut converter = AprConverter::new();
        converter.add_tensor(tensor);

        let apr_bytes = converter.to_apr().unwrap();

        // Should start with APR magic
        prop_assert_eq!(&apr_bytes[0..4], b"APRN");
    }

    /// Property: Element size is consistent with data type
    #[test]
    fn element_size_consistent(dtype in data_type()) {
        let expected = match dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::I8 | DataType::U8 | DataType::Q8_0 | DataType::Q4_0 => 1,
        };
        prop_assert_eq!(dtype.element_size(), expected);
    }

    /// Property: Format display is non-empty
    #[test]
    fn format_display_non_empty(
        format in prop_oneof![
            Just(ConversionFormat::SafeTensors),
            Just(ConversionFormat::Apr),
            Just(ConversionFormat::Gguf),
        ]
    ) {
        let display = format!("{}", format);
        prop_assert!(!display.is_empty());
    }

    /// Property: Format extension is lowercase
    #[test]
    fn format_extension_lowercase(
        format in prop_oneof![
            Just(ConversionFormat::SafeTensors),
            Just(ConversionFormat::Apr),
            Just(ConversionFormat::Gguf),
        ]
    ) {
        let ext = format.extension();
        prop_assert!(ext.chars().all(|c| c.is_lowercase() || c == '.'));
    }
}

#[cfg(test)]
mod conversion_support {
    use super::*;

    #[test]
    fn safetensors_to_apr_supported() {
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::SafeTensors,
            ConversionFormat::Apr
        ));
    }

    #[test]
    fn apr_to_gguf_supported() {
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Apr,
            ConversionFormat::Gguf
        ));
    }

    #[test]
    fn apr_to_apr_not_supported() {
        // APR->APR is not a conversion (identity)
        assert!(!AprConverter::is_conversion_supported(
            ConversionFormat::Apr,
            ConversionFormat::Apr
        ));
    }

    #[test]
    fn gguf_to_safetensors_not_supported() {
        assert!(!AprConverter::is_conversion_supported(
            ConversionFormat::Gguf,
            ConversionFormat::SafeTensors
        ));
    }
}

#[cfg(test)]
mod extension_parsing {
    use super::*;

    #[test]
    fn extension_roundtrip() {
        for format in [
            ConversionFormat::SafeTensors,
            ConversionFormat::Apr,
            ConversionFormat::Gguf,
        ] {
            let ext = format.extension();
            let parsed = ConversionFormat::from_extension(ext).unwrap();
            assert_eq!(parsed, format);
        }
    }

    #[test]
    fn extension_case_insensitive() {
        assert_eq!(
            ConversionFormat::from_extension("APR").unwrap(),
            ConversionFormat::Apr
        );
        assert_eq!(
            ConversionFormat::from_extension("SAFETENSORS").unwrap(),
            ConversionFormat::SafeTensors
        );
        assert_eq!(
            ConversionFormat::from_extension("GGUF").unwrap(),
            ConversionFormat::Gguf
        );
    }
}
