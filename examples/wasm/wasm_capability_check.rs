//! # WASM Capability Detector
//!
//! Browsers vary in WASM feature support: SIMD128 (Chrome 91+, Firefox
//! 89+, Safari 16.4+), threads (requires SAB + COOP/COEP headers),
//! bulk-memory (universal as of 2022). This recipe builds the
//! capability classifier + degradation tier.
//!
//! Demonstrates the **WASM.7** recipe for PMAT-123 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly proposals (https://github.com/WebAssembly/proposals)
//!
//! Run with: cargo run --example wasm_capability_check
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct WasmCapabilities {
    pub simd128: bool,
    pub threads: bool,
    pub bulk_memory: bool,
    pub reference_types: bool,
    pub multi_value: bool,
}

#[derive(Debug, PartialEq)]
pub enum FeatureTier {
    Modern,      // SIMD + threads + bulk + reftypes + multi-value
    Standard,    // bulk-memory + multi-value (2022 baseline)
    Legacy,      // MVP only
    Unsupported, // missing core requirement
}

pub fn classify(caps: WasmCapabilities) -> FeatureTier {
    if !caps.bulk_memory {
        return FeatureTier::Unsupported;
    }
    if caps.simd128 && caps.threads && caps.reference_types && caps.multi_value {
        return FeatureTier::Modern;
    }
    if caps.bulk_memory && caps.multi_value {
        return FeatureTier::Standard;
    }
    FeatureTier::Legacy
}

#[derive(Debug, PartialEq)]
pub enum ThreadsVerdict {
    Available,
    MissingSharedArrayBuffer,
    MissingCrossOriginIsolation,
    MissingThreadProposal,
}

pub fn check_threads(
    has_shared_array_buffer: bool,
    cross_origin_isolated: bool,
    threads_proposal: bool,
) -> ThreadsVerdict {
    if !threads_proposal {
        return ThreadsVerdict::MissingThreadProposal;
    }
    if !has_shared_array_buffer {
        return ThreadsVerdict::MissingSharedArrayBuffer;
    }
    if !cross_origin_isolated {
        return ThreadsVerdict::MissingCrossOriginIsolation;
    }
    ThreadsVerdict::Available
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_capability_check")?;

    let modern = WasmCapabilities {
        simd128: true,
        threads: true,
        bulk_memory: true,
        reference_types: true,
        multi_value: true,
    };
    let baseline = WasmCapabilities {
        bulk_memory: true,
        multi_value: true,
        ..Default::default()
    };
    let mvp = WasmCapabilities::default();

    println!("modern:   {:?}", classify(modern));
    println!("baseline: {:?}", classify(baseline));
    println!("mvp:      {:?}", classify(mvp));

    println!("threads (full): {:?}", check_threads(true, true, true));
    println!("threads (no COOP): {:?}", check_threads(true, false, true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modern() -> WasmCapabilities {
        WasmCapabilities {
            simd128: true,
            threads: true,
            bulk_memory: true,
            reference_types: true,
            multi_value: true,
        }
    }

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_caps_classified_modern() {
        assert_eq!(classify(modern()), FeatureTier::Modern);
    }

    #[test]
    fn no_bulk_memory_unsupported() {
        let mut c = modern();
        c.bulk_memory = false;
        assert_eq!(classify(c), FeatureTier::Unsupported);
    }

    #[test]
    fn baseline_2022_classified_standard() {
        let c = WasmCapabilities {
            bulk_memory: true,
            multi_value: true,
            ..Default::default()
        };
        assert_eq!(classify(c), FeatureTier::Standard);
    }

    #[test]
    fn missing_simd_drops_to_standard() {
        let mut c = modern();
        c.simd128 = false;
        assert_eq!(classify(c), FeatureTier::Standard);
    }

    #[test]
    fn missing_multi_value_drops_to_legacy() {
        let mut c = modern();
        c.multi_value = false;
        c.simd128 = false;
        c.threads = false;
        c.reference_types = false;
        assert_eq!(classify(c), FeatureTier::Legacy);
    }

    #[test]
    fn threads_full_stack_available() {
        assert_eq!(check_threads(true, true, true), ThreadsVerdict::Available);
    }

    #[test]
    fn threads_proposal_missing_rejected() {
        assert_eq!(
            check_threads(true, true, false),
            ThreadsVerdict::MissingThreadProposal
        );
    }

    #[test]
    fn threads_sab_missing_rejected() {
        assert_eq!(
            check_threads(false, true, true),
            ThreadsVerdict::MissingSharedArrayBuffer
        );
    }

    #[test]
    fn threads_no_coop_rejected() {
        // SAB requires Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy.
        assert_eq!(
            check_threads(true, false, true),
            ThreadsVerdict::MissingCrossOriginIsolation
        );
    }
}
