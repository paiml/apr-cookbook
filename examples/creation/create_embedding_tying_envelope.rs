//! # Creation Embedding-Output Tying Envelope
//!
//! Tied weights: the input embedding matrix W_e (vocab × hidden) and
//! output projection W_o (hidden × vocab) share the same parameters
//! (transposed). Saves vocab × hidden params; matters for small models
//! (10-30% of total). Constraint: requires identical shapes after
//! transpose. This recipe validates the tying decision.
//!
//! Demonstrates the **CREATE.9** recipe for PMAT-127 (creation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Press & Wolf (2017). Using the Output Embedding to Improve LMs. EACL.
//!
//! Run with: cargo run --example create_embedding_tying_envelope
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TyingVerdict {
    Ok {
        saved_params: u64,
    },
    ShapeMismatch {
        embed: (u32, u32),
        output: (u32, u32),
    },
    DtypeMismatch,
    InvalidShape,
}

pub fn validate(
    embed_shape: (u32, u32),
    output_shape: (u32, u32),
    dtypes_match: bool,
) -> TyingVerdict {
    if embed_shape.0 == 0 || embed_shape.1 == 0 || output_shape.0 == 0 || output_shape.1 == 0 {
        return TyingVerdict::InvalidShape;
    }
    if !dtypes_match {
        return TyingVerdict::DtypeMismatch;
    }
    // For tying, output (hidden × vocab) must equal embed transposed.
    if embed_shape.0 != output_shape.1 || embed_shape.1 != output_shape.0 {
        return TyingVerdict::ShapeMismatch {
            embed: embed_shape,
            output: output_shape,
        };
    }
    let saved = u64::from(embed_shape.0) * u64::from(embed_shape.1);
    TyingVerdict::Ok {
        saved_params: saved,
    }
}

pub fn pct_savings_of_total(saved_params: u64, total_params: u64) -> Option<f64> {
    if total_params == 0 {
        return None;
    }
    Some(saved_params as f64 / total_params as f64 * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("create_embedding_tying_envelope")?;

    let cases = [
        ((32_000u32, 4096u32), (4096u32, 32_000u32), true),
        ((32_000, 4096), (5120, 32_000), true), // shape mismatch
        ((32_000, 4096), (4096, 32_000), false), // dtype mismatch
        ((0, 4096), (4096, 0), true),           // invalid
    ];
    for (e, o, d) in cases {
        println!(
            "embed={e:?} output={o:?} dtypes_match={d}  →  {:?}",
            validate(e, o, d)
        );
    }
    println!(
        "saved 131M / 7B model: {:?}%",
        pct_savings_of_total(131_072_000, 7_000_000_000)
    );
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
    fn matched_shapes_and_dtype_passes() {
        let v = validate((32_000, 4096), (4096, 32_000), true);
        assert!(matches!(
            v,
            TyingVerdict::Ok {
                saved_params: 131_072_000
            }
        ));
    }

    #[test]
    fn shape_mismatch_rejected() {
        let v = validate((32_000, 4096), (5120, 32_000), true);
        assert!(matches!(v, TyingVerdict::ShapeMismatch { .. }));
    }

    #[test]
    fn vocab_mismatch_rejected() {
        let v = validate((32_000, 4096), (4096, 50_000), true);
        assert!(matches!(v, TyingVerdict::ShapeMismatch { .. }));
    }

    #[test]
    fn dtype_mismatch_rejected() {
        assert_eq!(
            validate((32_000, 4096), (4096, 32_000), false),
            TyingVerdict::DtypeMismatch
        );
    }

    #[test]
    fn zero_shape_invalid() {
        assert_eq!(
            validate((0, 4096), (4096, 0), true),
            TyingVerdict::InvalidShape
        );
    }

    #[test]
    fn saved_params_equals_embed_size() {
        if let TyingVerdict::Ok { saved_params } = validate((1000, 128), (128, 1000), true) {
            assert_eq!(saved_params, 128_000);
        }
    }

    #[test]
    fn percent_savings_basic() {
        // 100K saved / 1M total = 10%.
        let pct = pct_savings_of_total(100_000, 1_000_000).unwrap();
        assert!((pct - 10.0).abs() < 1e-9);
    }

    #[test]
    fn percent_savings_zero_total_invalid() {
        assert!(pct_savings_of_total(100, 0).is_none());
    }

    #[test]
    fn small_model_high_savings_pct() {
        // 50M embed in 200M model → 25%.
        let pct = pct_savings_of_total(50_000_000, 200_000_000).unwrap();
        assert!(pct > 20.0);
    }

    #[test]
    fn large_model_low_savings_pct() {
        // 131M embed in 7B model → < 2%.
        let pct = pct_savings_of_total(131_072_000, 7_000_000_000).unwrap();
        assert!(pct < 2.0);
    }
}
