//! # apr stamp — SPDX License Validation
//!
//! `apr stamp` enforces SPDX validity for the `--license` and `--data-license`
//! fields. The accepted set is the SPDX License List 3.x identifier vocabulary
//! plus the explicit `NOASSERTION` token. A typo like `Apache2` or a freeform
//! license blurb is rejected at the CLI boundary so it can never reach the APR
//! v2 provenance section.
//!
//! Demonstrates the **STAMP.3** recipe for PMAT-088 (apr stamp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SPDX License List 3.x (spdx.org/licenses) + aprender SHIP-009
//!
//! Run with: cargo run --example cli_stamp_spdx_validation
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

/// Minimal SPDX 3.x allowlist for cookbook recipes. The real `apr stamp`
/// links the full SPDX License List; we vendor a representative subset so
/// the recipe stays offline-only.
const SPDX_ALLOWED: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MPL-2.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-3.0-only",
    "ISC",
    "Unlicense",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "NOASSERTION",
];

#[derive(Debug, PartialEq)]
enum SpdxVerdict {
    Accepted,
    Rejected(&'static str),
}

fn validate_spdx(license: &str) -> SpdxVerdict {
    if license.is_empty() {
        return SpdxVerdict::Rejected("empty SPDX identifier");
    }
    if license.contains(char::is_whitespace) {
        return SpdxVerdict::Rejected("SPDX identifiers contain no whitespace");
    }
    if SPDX_ALLOWED.contains(&license) {
        SpdxVerdict::Accepted
    } else {
        SpdxVerdict::Rejected("not in SPDX License List 3.x allowlist")
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_stamp_spdx_validation")?;

    let candidates = [
        "Apache-2.0",
        "MIT",
        "Apache2",
        "Apache 2",
        "Released under Apache 2 with caveats",
        "NOASSERTION",
        "",
    ];

    for c in candidates {
        let verdict = validate_spdx(c);
        println!("{c:>50}  →  {verdict:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn canonical_spdx_accepted() {
        assert_eq!(validate_spdx("Apache-2.0"), SpdxVerdict::Accepted);
        assert_eq!(validate_spdx("MIT"), SpdxVerdict::Accepted);
        assert_eq!(validate_spdx("CC-BY-SA-4.0"), SpdxVerdict::Accepted);
    }

    #[test]
    fn noassertion_accepted() {
        // Per SHIP-009: NOASSERTION is the explicit "I haven't decided" token,
        // not the same as missing/empty.
        assert_eq!(validate_spdx("NOASSERTION"), SpdxVerdict::Accepted);
    }

    #[test]
    fn typo_rejected() {
        // Apache2 is the most common SPDX typo seen in the wild.
        assert!(matches!(validate_spdx("Apache2"), SpdxVerdict::Rejected(_)));
    }

    #[test]
    fn whitespace_rejected() {
        assert!(matches!(
            validate_spdx("Apache 2"),
            SpdxVerdict::Rejected(_)
        ));
    }

    #[test]
    fn freeform_rejected() {
        // Freeform license blurbs must never land in provenance — provenance
        // tooling depends on parseable SPDX identifiers.
        assert!(matches!(
            validate_spdx("Released under Apache 2 with caveats"),
            SpdxVerdict::Rejected(_)
        ));
    }

    #[test]
    fn empty_rejected() {
        assert!(matches!(validate_spdx(""), SpdxVerdict::Rejected(_)));
    }
}
