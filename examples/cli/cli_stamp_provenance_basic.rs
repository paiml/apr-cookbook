//! # apr stamp — Basic Provenance Stamping
//!
//! `apr stamp --license <SPDX> --data-source <URL> --data-license <SPDX>
//! --output <NEW.apr> <IN.apr>` patches the three provenance fields on a
//! pre-built APR v2 artifact (the SHIP-009 full-discharge enabler for the
//! MODEL-1 teacher whose fields were `(missing)` because it shipped before
//! GATE-APR-PROV-001..003).
//!
//! Tensor bytes and header flags are preserved verbatim — only the three
//! provenance fields change. This recipe demonstrates building the
//! invocation envelope and asserting the schema (all 3 fields are required
//! for a complete stamp; partial stamps are allowed).
//!
//! Demonstrates the **STAMP.1** recipe for PMAT-088 (apr stamp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + GATE-APR-PROV-001..003 + SPDX License List 3.x
//!
//! Run with: cargo run --example cli_stamp_provenance_basic
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
struct StampInvocation {
    input_path: String,
    output_path: String,
    license: Option<String>,
    data_source: Option<String>,
    data_license: Option<String>,
}

fn build_stamp_invocation(
    input: &str,
    output: &str,
    license: Option<&str>,
    data_source: Option<&str>,
    data_license: Option<&str>,
) -> Result<StampInvocation> {
    if input == output {
        return Err(apr_cookbook::CookbookError::Validation(
            "input and output paths must differ (in-place stamp not supported)".into(),
        ));
    }
    if license.is_none() && data_source.is_none() && data_license.is_none() {
        return Err(apr_cookbook::CookbookError::Validation(
            "at least one of --license, --data-source, --data-license must be provided".into(),
        ));
    }
    Ok(StampInvocation {
        input_path: input.into(),
        output_path: output.into(),
        license: license.map(String::from),
        data_source: data_source.map(String::from),
        data_license: data_license.map(String::from),
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_stamp_provenance_basic")?;

    let inv = build_stamp_invocation(
        "model.apr",
        "model-stamped.apr",
        Some("Apache-2.0"),
        Some("huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"),
        Some("Apache-2.0"),
    )?;

    println!("apr stamp --license {license:?} --data-source {ds:?} --data-license {dl:?} --output {out} {input}",
        license = inv.license,
        ds = inv.data_source,
        dl = inv.data_license,
        out = inv.output_path,
        input = inv.input_path);
    println!(
        "\n(in real `apr stamp`, this patches the provenance fields and writes a new APR v2 file)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_invocation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn in_place_stamp_rejected() {
        // Tensor bytes are preserved verbatim, but the writer needs a different
        // output path because the provenance update changes the header.
        let err = build_stamp_invocation("a.apr", "a.apr", Some("MIT"), None, None);
        assert!(err.is_err());
    }

    #[test]
    fn empty_stamp_rejected() {
        let err = build_stamp_invocation("a.apr", "b.apr", None, None, None);
        assert!(err.is_err());
    }

    #[test]
    fn partial_stamp_allowed() {
        // SHIP-009 supports stamping individual fields; full triple is not required.
        let inv = build_stamp_invocation("a.apr", "b.apr", Some("MIT"), None, None).unwrap();
        assert_eq!(inv.license, Some("MIT".into()));
        assert_eq!(inv.data_source, None);
    }
}
