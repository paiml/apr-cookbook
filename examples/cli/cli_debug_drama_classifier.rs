//! # apr debug — `--drama` Mode Classifier
//!
//! `apr debug --drama <FILE>` enables theatrical output for the model
//! file — file size, dtype, and tensor count get a melodramatic
//! description. This recipe codifies the categorisation as a pure
//! function so the message changes can be tracked over time.
//!
//! Demonstrates the **DEBUG.5** recipe for PMAT-101 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEBUG-002 (drama mode)
//!
//! Run with: cargo run --example cli_debug_drama_classifier
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeDrama {
    Tiny,     // < 100 MB
    Small,    // < 1 GB
    Beefy,    // < 10 GB
    Hefty,    // < 50 GB
    Behemoth, // ≥ 50 GB
}

pub fn classify_size(bytes: u64) -> SizeDrama {
    const MB: u64 = 1_000_000;
    const GB: u64 = 1_000_000_000;
    match bytes {
        n if n < 100 * MB => SizeDrama::Tiny,
        n if n < GB => SizeDrama::Small,
        n if n < 10 * GB => SizeDrama::Beefy,
        n if n < 50 * GB => SizeDrama::Hefty,
        _ => SizeDrama::Behemoth,
    }
}

pub fn drama_message(size: SizeDrama, dtype: &str, tensor_count: usize) -> String {
    let size_phrase = match size {
        SizeDrama::Tiny => "fits in your pocket",
        SizeDrama::Small => "fits in RAM, no sweat",
        SizeDrama::Beefy => "wants its own GPU",
        SizeDrama::Hefty => "demands your A100",
        SizeDrama::Behemoth => "walks like a giant",
    };
    let dtype_phrase = match dtype {
        "fp32" => "in full glory",
        "fp16" | "bf16" => "with mixed precision",
        "int8" => "compressed to 8 bits",
        "int4" => "squeezed to a quarter byte",
        _ => "in some unfamiliar dtype",
    };
    format!("This {tensor_count}-tensor model {size_phrase}, {dtype_phrase}.")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_drama_classifier")?;

    let cases = [
        ("tiny bert", 50_000_000, "fp32", 200),
        ("medium qwen", 7_000_000_000, "bf16", 350),
        ("hefty llama", 14_000_000_000, "int8", 280),
        ("behemoth dense", 200_000_000_000u64, "bf16", 9000),
    ];

    for (label, b, dt, n) in cases {
        let size = classify_size(b);
        println!("{label:>15}  →  {}", drama_message(size, dt, n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drama_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_classifies_correctly() {
        assert_eq!(classify_size(50_000_000), SizeDrama::Tiny);
        assert_eq!(classify_size(500_000_000), SizeDrama::Small);
    }

    #[test]
    fn boundaries_classify_to_higher_band() {
        // 100 MB exactly → Small (not Tiny).
        assert_eq!(classify_size(100_000_000), SizeDrama::Small);
        // 1 GB exactly → Beefy.
        assert_eq!(classify_size(1_000_000_000), SizeDrama::Beefy);
        // 10 GB exactly → Hefty.
        assert_eq!(classify_size(10_000_000_000), SizeDrama::Hefty);
        // 50 GB exactly → Behemoth.
        assert_eq!(classify_size(50_000_000_000), SizeDrama::Behemoth);
    }

    #[test]
    fn drama_message_includes_size_phrase() {
        let m = drama_message(SizeDrama::Tiny, "fp32", 100);
        assert!(m.contains("fits in your pocket"));
    }

    #[test]
    fn drama_message_includes_dtype_phrase() {
        let m = drama_message(SizeDrama::Beefy, "int4", 500);
        assert!(m.contains("squeezed to a quarter byte"));
    }

    #[test]
    fn drama_message_includes_tensor_count() {
        let m = drama_message(SizeDrama::Tiny, "fp32", 42);
        assert!(m.contains("42-tensor"));
    }

    #[test]
    fn unknown_dtype_falls_through_to_safe_phrase() {
        // Doesn't panic on weird dtypes — just uses a generic phrase.
        let m = drama_message(SizeDrama::Tiny, "qx_qy", 1);
        assert!(m.contains("unfamiliar dtype"));
    }
}
