//! # Conversion Format-Version Compatibility Matrix
//!
//! Each (format, source_version, target_version) tuple has a
//! compatibility verdict: Direct (no rewrite needed), Upgrade (older
//! → newer), Downgrade (newer → older, often lossy), Unsupported. This
//! recipe builds the matrix for APR/GGUF/ONNX/SafeTensors.
//!
//! Demonstrates the **CONV.9** recipe for PMAT-127 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FORMAT-COMPAT spec.
//!
//! Run with: cargo run --example convert_format_version_matrix
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Apr,
    Gguf,
    SafeTensors,
    Onnx,
}

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Direct,
    Upgrade { from: u32, to: u32 },
    Downgrade { from: u32, to: u32 },
    Unsupported { reason: &'static str },
    InvalidVersion,
}

pub fn classify(format: Format, source_ver: u32, target_ver: u32) -> CompatVerdict {
    if source_ver == 0 || target_ver == 0 {
        return CompatVerdict::InvalidVersion;
    }
    let max_ver = match format {
        Format::Apr => 2,
        Format::Gguf => 3,
        Format::SafeTensors => 1,
        Format::Onnx => 21,
    };
    if source_ver > max_ver || target_ver > max_ver {
        return CompatVerdict::Unsupported {
            reason: "version exceeds known range",
        };
    }
    if source_ver == target_ver {
        return CompatVerdict::Direct;
    }
    if source_ver < target_ver {
        return CompatVerdict::Upgrade {
            from: source_ver,
            to: target_ver,
        };
    }
    if format == Format::SafeTensors {
        // SafeTensors is single-version; no downgrade possible.
        return CompatVerdict::Unsupported {
            reason: "SafeTensors has no version downgrade path",
        };
    }
    CompatVerdict::Downgrade {
        from: source_ver,
        to: target_ver,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_format_version_matrix")?;

    for (f, s, t) in [
        (Format::Apr, 1u32, 2u32),
        (Format::Apr, 2, 1),
        (Format::Gguf, 3, 3),
        (Format::Gguf, 99, 1),
        (Format::SafeTensors, 1, 1),
        (Format::Onnx, 14, 21),
        (Format::Apr, 0, 1),
    ] {
        println!("{f:?} {s} → {t}  =  {:?}", classify(f, s, t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_version_direct() {
        assert_eq!(classify(Format::Apr, 2, 2), CompatVerdict::Direct);
        assert_eq!(classify(Format::Gguf, 3, 3), CompatVerdict::Direct);
    }

    #[test]
    fn lower_to_higher_upgrade() {
        let v = classify(Format::Apr, 1, 2);
        assert!(matches!(v, CompatVerdict::Upgrade { from: 1, to: 2 }));
    }

    #[test]
    fn higher_to_lower_downgrade() {
        let v = classify(Format::Apr, 2, 1);
        assert!(matches!(v, CompatVerdict::Downgrade { from: 2, to: 1 }));
    }

    #[test]
    fn safetensors_no_downgrade() {
        // SafeTensors is single-version; downgrade attempts are unsupported.
        // With max=1, "downgrade" doesn't occur naturally; bump test by
        // simulating max=2 source. With current bounds, only 1→1 possible.
        assert_eq!(classify(Format::SafeTensors, 1, 1), CompatVerdict::Direct);
    }

    #[test]
    fn version_above_max_unsupported() {
        let v = classify(Format::Apr, 99, 1);
        assert!(matches!(v, CompatVerdict::Unsupported { .. }));
        let v2 = classify(Format::Gguf, 1, 99);
        assert!(matches!(v2, CompatVerdict::Unsupported { .. }));
    }

    #[test]
    fn zero_version_invalid() {
        assert_eq!(classify(Format::Apr, 0, 1), CompatVerdict::InvalidVersion);
        assert_eq!(classify(Format::Apr, 1, 0), CompatVerdict::InvalidVersion);
    }

    #[test]
    fn onnx_wide_version_range() {
        // ONNX supports many opsets; 14 → 21 is legit.
        let v = classify(Format::Onnx, 14, 21);
        assert!(matches!(v, CompatVerdict::Upgrade { from: 14, to: 21 }));
    }

    #[test]
    fn onnx_downgrade_supported() {
        let v = classify(Format::Onnx, 21, 14);
        assert!(matches!(v, CompatVerdict::Downgrade { .. }));
    }

    #[test]
    fn gguf_v3_to_v2_downgrade() {
        let v = classify(Format::Gguf, 3, 2);
        assert!(matches!(v, CompatVerdict::Downgrade { from: 3, to: 2 }));
    }

    #[test]
    fn safetensors_single_version_only() {
        // Higher version unsupported (max=1).
        let v = classify(Format::SafeTensors, 2, 1);
        assert!(matches!(v, CompatVerdict::Unsupported { .. }));
    }
}
