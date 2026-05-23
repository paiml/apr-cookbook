//! # Conversion — GGUF Q4_0/Q5_0/Q8_0 Import Fallback (GH-375)
//!
//! aprender GH-375 added a dequant-requant fallback for legacy GGUF
//! quantization types (Q4_0, Q5_0, Q8_0) that aren't supported by the
//! direct import path. The fallback path: GGUF → f32 intermediate → Q4_K
//! (the modern superblock format). Raw import preserves Q4_K/Q6_K
//! exactly; legacy types go through this f32 dequant-requant.
//!
//! This recipe demonstrates the dispatch logic: classify quantization
//! type as native (raw-import) vs legacy (dequant-requant), and apply
//! the right path.
//!
//! Demonstrates the **CV+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-375 + ggerganov/llama.cpp GGUF spec
//!
//! Run with: cargo run --example conversion_gguf_legacy_quant_fallback
//!
//! Added by PMAT-085 (expand-cookbooks: Tier 3 perf benches).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufQuant {
    Q4K,
    Q6K,
    Q4_0,
    Q5_0,
    Q8_0,
    F16,
    F32,
}

#[derive(Debug, PartialEq, Eq)]
enum ImportPath {
    Native,
    DequantRequant,
}

fn classify_import_path(quant: GgufQuant) -> ImportPath {
    match quant {
        // Modern superblock formats: native (raw) import preserves bits exactly.
        GgufQuant::Q4K | GgufQuant::Q6K => ImportPath::Native,
        // Legacy formats: dequant to f32 intermediate, then re-quant to Q4K.
        GgufQuant::Q4_0 | GgufQuant::Q5_0 | GgufQuant::Q8_0 | GgufQuant::F16 | GgufQuant::F32 => {
            ImportPath::DequantRequant
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("conversion_gguf_legacy_quant_fallback")?;
    let cases = [
        GgufQuant::Q4K,
        GgufQuant::Q6K,
        GgufQuant::Q4_0,
        GgufQuant::Q5_0,
        GgufQuant::Q8_0,
        GgufQuant::F16,
        GgufQuant::F32,
    ];
    println!("GGUF import path dispatch (per GH-375):");
    for q in &cases {
        let p = classify_import_path(*q);
        println!("  {q:?} -> {p:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn modern_superblocks_use_native_path() {
        assert_eq!(classify_import_path(GgufQuant::Q4K), ImportPath::Native);
        assert_eq!(classify_import_path(GgufQuant::Q6K), ImportPath::Native);
    }

    #[test]
    fn legacy_quants_use_dequant_requant_path() {
        for q in [GgufQuant::Q4_0, GgufQuant::Q5_0, GgufQuant::Q8_0] {
            assert_eq!(classify_import_path(q), ImportPath::DequantRequant);
        }
    }
}
