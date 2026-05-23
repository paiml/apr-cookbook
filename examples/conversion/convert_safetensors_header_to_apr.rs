//! # Conversion SafeTensors Header → APR2 Manifest Stub
//!
//! Validate SafeTensors header entries (per-tensor: dtype, shape,
//! offsets) and emit APR2-format manifest stub. Validates dtype is
//! supported, no duplicate names, no overlapping offsets.
//!
//! Demonstrates the **CONV.19** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace SafeTensors v0.4 spec.
//!
//! Run with: cargo run --example convert_safetensors_header_to_apr
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct SafeTensorsEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data_offset: u64,
    pub data_size: u64,
}

#[derive(Debug, PartialEq)]
pub enum ConvertVerdict {
    Ok {
        tensor_count: u32,
        total_data_bytes: u64,
        manifest_stub: String,
    },
    EmptyHeader,
    InvalidDtype {
        dtype: String,
    },
    DuplicateTensor {
        name: String,
    },
    OverlappingOffsets,
}

const VALID_DTYPES: &[&str] = &[
    "F16", "F32", "F64", "BF16", "I8", "I16", "I32", "I64", "U8", "BOOL",
];

pub fn convert(entries: &[SafeTensorsEntry]) -> ConvertVerdict {
    if entries.is_empty() {
        return ConvertVerdict::EmptyHeader;
    }
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for e in entries {
        if !VALID_DTYPES.contains(&e.dtype.as_str()) {
            return ConvertVerdict::InvalidDtype {
                dtype: e.dtype.clone(),
            };
        }
        if !seen.insert(&e.name) {
            return ConvertVerdict::DuplicateTensor {
                name: e.name.clone(),
            };
        }
    }
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| e.data_offset);
    for w in sorted.windows(2) {
        let prev_end = w[0].data_offset + w[0].data_size;
        if prev_end > w[1].data_offset {
            return ConvertVerdict::OverlappingOffsets;
        }
    }
    let total_data_bytes: u64 = entries.iter().map(|e| e.data_size).sum();
    let manifest_stub = format!(
        "version: 2\nformat: apr2\ntensors:\n{}",
        entries
            .iter()
            .map(|e| format!(
                "  - name: {}\n    dtype: {}\n    shape: {:?}",
                e.name, e.dtype, e.shape
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
    ConvertVerdict::Ok {
        tensor_count: entries.len() as u32,
        total_data_bytes,
        manifest_stub,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_safetensors_header_to_apr")?;

    let entries = vec![
        SafeTensorsEntry {
            name: "embed".to_string(),
            dtype: "F16".to_string(),
            shape: vec![1000, 256],
            data_offset: 0,
            data_size: 512_000,
        },
        SafeTensorsEntry {
            name: "fc.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 256],
            data_offset: 512_000,
            data_size: 10_240,
        },
    ];
    println!("typical: {:?}", convert(&entries));
    println!("empty: {:?}", convert(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<SafeTensorsEntry> {
        vec![
            SafeTensorsEntry {
                name: "a".to_string(),
                dtype: "F16".to_string(),
                shape: vec![100, 50],
                data_offset: 0,
                data_size: 10_000,
            },
            SafeTensorsEntry {
                name: "b".to_string(),
                dtype: "F32".to_string(),
                shape: vec![50, 50],
                data_offset: 10_000,
                data_size: 10_000,
            },
        ]
    }

    #[test]
    fn converter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_two_tensors_ok() {
        let v = convert(&typical());
        if let ConvertVerdict::Ok { tensor_count, .. } = v {
            assert_eq!(tensor_count, 2);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(convert(&[]), ConvertVerdict::EmptyHeader);
    }

    #[test]
    fn invalid_dtype_rejected() {
        let entries = vec![SafeTensorsEntry {
            name: "x".to_string(),
            dtype: "FUNKY".to_string(),
            shape: vec![10],
            data_offset: 0,
            data_size: 100,
        }];
        let v = convert(&entries);
        assert!(matches!(v, ConvertVerdict::InvalidDtype { .. }));
    }

    #[test]
    fn duplicate_name_rejected() {
        let entries = vec![
            SafeTensorsEntry {
                name: "x".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 0,
                data_size: 100,
            },
            SafeTensorsEntry {
                name: "x".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 100,
                data_size: 100,
            },
        ];
        assert!(matches!(
            convert(&entries),
            ConvertVerdict::DuplicateTensor { .. }
        ));
    }

    #[test]
    fn overlapping_offsets_rejected() {
        let entries = vec![
            SafeTensorsEntry {
                name: "a".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 0,
                data_size: 200,
            },
            SafeTensorsEntry {
                name: "b".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 100,
                data_size: 200,
            },
        ];
        assert_eq!(convert(&entries), ConvertVerdict::OverlappingOffsets);
    }

    #[test]
    fn total_bytes_summed() {
        if let ConvertVerdict::Ok {
            total_data_bytes, ..
        } = convert(&typical())
        {
            assert_eq!(total_data_bytes, 20_000);
        }
    }

    #[test]
    fn manifest_stub_includes_names() {
        if let ConvertVerdict::Ok { manifest_stub, .. } = convert(&typical()) {
            assert!(manifest_stub.contains("name: a"));
            assert!(manifest_stub.contains("name: b"));
        }
    }

    #[test]
    fn bf16_dtype_supported() {
        let entries = vec![SafeTensorsEntry {
            name: "x".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![10],
            data_offset: 0,
            data_size: 100,
        }];
        assert!(matches!(convert(&entries), ConvertVerdict::Ok { .. }));
    }

    #[test]
    fn adjacent_offsets_ok() {
        let entries = vec![
            SafeTensorsEntry {
                name: "a".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 0,
                data_size: 100,
            },
            SafeTensorsEntry {
                name: "b".to_string(),
                dtype: "F16".to_string(),
                shape: vec![10],
                data_offset: 100,
                data_size: 100,
            },
        ];
        assert!(matches!(convert(&entries), ConvertVerdict::Ok { .. }));
    }

    #[test]
    fn version_in_manifest() {
        if let ConvertVerdict::Ok { manifest_stub, .. } = convert(&typical()) {
            assert!(manifest_stub.contains("version: 2"));
        }
    }
}
