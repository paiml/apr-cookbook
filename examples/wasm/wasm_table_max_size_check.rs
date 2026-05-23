//! # WASM Table Max Size Check
//!
//! Validate WASM table declaration: initial ≤ max ≤ engine limit
//! (typically 10M for funcref tables). Returns categorical verdict.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §3.2.5 table types; V8 wasm-engine
//!  table-size enforcement.
//!
//! Run with: cargo run --example wasm_table_max_size_check
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TableSizeVerdict {
    Valid,
    InitialOverMax,
    MaxOverEngineLimit,
    InvalidConfig,
}

pub fn validate(initial: u32, max: Option<u32>, engine_limit: u32) -> TableSizeVerdict {
    if engine_limit == 0 {
        return TableSizeVerdict::InvalidConfig;
    }
    if let Some(m) = max {
        if initial > m {
            return TableSizeVerdict::InitialOverMax;
        }
        if m > engine_limit {
            return TableSizeVerdict::MaxOverEngineLimit;
        }
    } else if initial > engine_limit {
        return TableSizeVerdict::MaxOverEngineLimit;
    }
    TableSizeVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_table_max_size_check")?;

    println!("ok: {:?}", validate(10, Some(100), 10_000));
    println!("init>max: {:?}", validate(200, Some(100), 10_000));
    println!("max>limit: {:?}", validate(0, Some(100_000), 10_000));
    println!("invalid: {:?}", validate(0, None, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_bounds_valid() {
        assert_eq!(validate(10, Some(100), 10_000), TableSizeVerdict::Valid);
    }

    #[test]
    fn initial_over_max_rejected() {
        assert_eq!(
            validate(200, Some(100), 10_000),
            TableSizeVerdict::InitialOverMax
        );
    }

    #[test]
    fn max_over_engine_rejected() {
        assert_eq!(
            validate(0, Some(100_000), 10_000),
            TableSizeVerdict::MaxOverEngineLimit
        );
    }

    #[test]
    fn no_max_initial_within_limit_valid() {
        assert_eq!(validate(100, None, 10_000), TableSizeVerdict::Valid);
    }

    #[test]
    fn no_max_initial_over_limit_rejected() {
        assert_eq!(
            validate(100_000, None, 10_000),
            TableSizeVerdict::MaxOverEngineLimit
        );
    }

    #[test]
    fn zero_engine_limit_rejected() {
        assert_eq!(validate(0, None, 0), TableSizeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = validate(10, Some(100), 10_000);
        let r2 = validate(10, Some(100), 10_000);
        assert_eq!(r1, r2);
    }

    #[test]
    fn initial_eq_max_valid() {
        assert_eq!(validate(100, Some(100), 10_000), TableSizeVerdict::Valid);
    }

    #[test]
    fn max_eq_engine_limit_valid() {
        assert_eq!(validate(0, Some(10_000), 10_000), TableSizeVerdict::Valid);
    }

    #[test]
    fn zero_initial_no_max_valid() {
        assert_eq!(validate(0, None, 100), TableSizeVerdict::Valid);
    }

    #[test]
    fn zero_initial_with_max_valid() {
        assert_eq!(validate(0, Some(10), 100), TableSizeVerdict::Valid);
    }

    #[test]
    fn high_engine_limit_handled() {
        assert_eq!(
            validate(1000, Some(10_000), u32::MAX),
            TableSizeVerdict::Valid
        );
    }
}
