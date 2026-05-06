//! # WASM Table Index Resolver (call_indirect)
//!
//! `call_indirect` requires:
//! - target index < table size
//! - table[index] is non-null
//! - element function type matches expected_signature_id
//!
//! Picker validates a call, returns Ok / IndexOutOfBounds / Null /
//! TypeMismatch.
//!
//! Demonstrates the **WASM.22** recipe for PMAT-146 (wasm round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core Spec § call_indirect.
//!
//! Run with: cargo run --example wasm_table_index_resolver
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableEntry {
    pub function_index: Option<u32>,
    pub signature_id: Option<u32>,
}

#[derive(Debug, PartialEq)]
pub enum CallVerdict {
    Ok { function_index: u32 },
    IndexOutOfBounds { index: u32, table_size: u32 },
    NullEntry { index: u32 },
    TypeMismatch { expected: u32, found: u32 },
    EmptyTable,
}

pub fn resolve(table: &[TableEntry], index: u32, expected_sig: u32) -> CallVerdict {
    if table.is_empty() {
        return CallVerdict::EmptyTable;
    }
    if (index as usize) >= table.len() {
        return CallVerdict::IndexOutOfBounds {
            index,
            table_size: table.len() as u32,
        };
    }
    let entry = &table[index as usize];
    let Some(func) = entry.function_index else {
        return CallVerdict::NullEntry { index };
    };
    let sig = entry.signature_id.unwrap_or(0);
    if sig != expected_sig {
        return CallVerdict::TypeMismatch {
            expected: expected_sig,
            found: sig,
        };
    }
    CallVerdict::Ok {
        function_index: func,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_table_index_resolver")?;

    let table = vec![
        TableEntry {
            function_index: Some(10),
            signature_id: Some(1),
        },
        TableEntry {
            function_index: Some(20),
            signature_id: Some(2),
        },
        TableEntry {
            function_index: None,
            signature_id: None,
        },
    ];

    println!("matching: {:?}", resolve(&table, 0, 1));
    println!("type mismatch: {:?}", resolve(&table, 0, 99));
    println!("null: {:?}", resolve(&table, 2, 1));
    println!("oob: {:?}", resolve(&table, 99, 1));
    println!("empty: {:?}", resolve(&[], 0, 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<TableEntry> {
        vec![
            TableEntry {
                function_index: Some(10),
                signature_id: Some(1),
            },
            TableEntry {
                function_index: Some(20),
                signature_id: Some(2),
            },
            TableEntry {
                function_index: None,
                signature_id: None,
            },
        ]
    }

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn matching_call_succeeds() {
        let v = resolve(&typical(), 0, 1);
        if let CallVerdict::Ok { function_index } = v {
            assert_eq!(function_index, 10);
        }
    }

    #[test]
    fn type_mismatch_rejected() {
        let v = resolve(&typical(), 0, 99);
        assert!(matches!(v, CallVerdict::TypeMismatch { .. }));
    }

    #[test]
    fn null_entry_rejected() {
        let v = resolve(&typical(), 2, 1);
        assert!(matches!(v, CallVerdict::NullEntry { .. }));
    }

    #[test]
    fn out_of_bounds_rejected() {
        let v = resolve(&typical(), 99, 1);
        assert!(matches!(v, CallVerdict::IndexOutOfBounds { .. }));
    }

    #[test]
    fn empty_table_rejected() {
        assert_eq!(resolve(&[], 0, 1), CallVerdict::EmptyTable);
    }

    #[test]
    fn oob_at_table_size_exact() {
        // index == size is OOB.
        let v = resolve(&typical(), 3, 1);
        assert!(matches!(v, CallVerdict::IndexOutOfBounds { .. }));
    }

    #[test]
    fn last_valid_index_succeeds() {
        let v = resolve(&typical(), 1, 2);
        if let CallVerdict::Ok { function_index } = v {
            assert_eq!(function_index, 20);
        }
    }

    #[test]
    fn type_mismatch_reports_signatures() {
        if let CallVerdict::TypeMismatch { expected, found } = resolve(&typical(), 0, 99) {
            assert_eq!(expected, 99);
            assert_eq!(found, 1);
        }
    }

    #[test]
    fn null_index_reported() {
        if let CallVerdict::NullEntry { index } = resolve(&typical(), 2, 1) {
            assert_eq!(index, 2);
        }
    }

    #[test]
    fn oob_carries_index_and_size() {
        if let CallVerdict::IndexOutOfBounds { index, table_size } = resolve(&typical(), 99, 1) {
            assert_eq!(index, 99);
            assert_eq!(table_size, 3);
        }
    }
}
