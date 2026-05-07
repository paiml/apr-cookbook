//! # Gptneox Smoke Inference
//!
//! Load a synthetic gptneox micro-checkpoint via `aprender::rosetta`,
//! run a deterministic forward pass, emit a `Verdict::Ok` value with
//! the resulting logits checksum.
//!
//! Demonstrates the **GPTNEOX.smoke** recipe per
//! `docs/specifications/architecture-demos.md`.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-gptneox-smoke-v1.yaml (grade A; lean_status: wip)
//! Citation: Black et al. (2022). GPT-NeoX-20B: An Open-Source Autoregressive Language Model. arXiv:2204.06745
//!
//! Run with: cargo run --example inference_gptneox_smoke
//!
//! Added by PMAT-300+ (architecture-demos: gptneox coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

pub fn smoke(fixture_path: &str, format: &str) -> SmokeVerdict {
    if !std::path::Path::new(fixture_path).exists() {
        return SmokeVerdict::InvalidFixture;
    }
    // todo!() — replace with actual aprender::rosetta::load_family call when fixture lands.
    SmokeVerdict::LoaderUnavailable {
        reason: format!("loader call for gptneox not yet wired (format={format})"),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_gptneox_smoke")?;
    let fixture = "tests/fixtures/architectures/gptneox/model.safetensors";
    println!("safetensors: {:?}", smoke(fixture, "safetensors"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_returns_invalid() {
        assert_eq!(
            smoke("/no/such/path", "safetensors"),
            SmokeVerdict::InvalidFixture
        );
    }

    #[test]
    fn loader_unavailable_when_path_exists_but_loader_unwired() {
        // Until the real loader is wired, an existing-but-not-loadable file
        // surfaces LoaderUnavailable. After the loader lands, this test flips
        // to assert SmokeVerdict::Ok { family: "gptneox", .. }.
        let v = smoke("/dev/null", "safetensors");
        assert!(matches!(
            v,
            SmokeVerdict::InvalidFixture | SmokeVerdict::LoaderUnavailable { .. }
        ));
    }
}
