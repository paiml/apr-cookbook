//! # apr decrypt — Invocation Envelope
//!
//! `apr decrypt --output <FILE> --key-file <KEY> [--force] <FILE.enc>`
//! decrypts AES-256-GCM-encrypted models. This recipe models the
//! invocation envelope as a pure function so a CI pipeline can preview
//! which combinations would be rejected at the boundary (missing key,
//! same input/output path, force-overwrite of existing file).
//!
//! Demonstrates the **DECRYPT.3** recipe for PMAT-095 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + AES-256-GCM (NIST SP 800-38D)
//!
//! Run with: cargo run --example cli_decrypt_invocation_envelope
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone)]
pub struct DecryptInvocation {
    pub input: String,
    pub output: String,
    pub key_file: Option<String>,
    pub passphrase_via_stdin: bool,
    pub force: bool,
    pub output_exists: bool,
}

#[derive(Debug, PartialEq)]
pub enum InvocationVerdict {
    Ok,
    InvalidPaths, // input == output, or empty
    NoKeySource,  // neither --key-file nor stdin
    OutputExistsWithoutForce,
}

pub fn validate_invocation(inv: &DecryptInvocation) -> InvocationVerdict {
    if inv.input.is_empty() || inv.output.is_empty() || inv.input == inv.output {
        return InvocationVerdict::InvalidPaths;
    }
    if inv.key_file.is_none() && !inv.passphrase_via_stdin {
        return InvocationVerdict::NoKeySource;
    }
    if inv.output_exists && !inv.force {
        return InvocationVerdict::OutputExistsWithoutForce;
    }
    InvocationVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_invocation_envelope")?;

    let cases: &[(&str, DecryptInvocation)] = &[
        (
            "happy w/ key",
            DecryptInvocation {
                input: "model.apr.enc".into(),
                output: "model.apr".into(),
                key_file: Some("decrypt.key".into()),
                ..Default::default()
            },
        ),
        (
            "stdin passphrase",
            DecryptInvocation {
                input: "model.apr.enc".into(),
                output: "model.apr".into(),
                passphrase_via_stdin: true,
                ..Default::default()
            },
        ),
        (
            "no key source",
            DecryptInvocation {
                input: "model.apr.enc".into(),
                output: "model.apr".into(),
                ..Default::default()
            },
        ),
        (
            "exists no force",
            DecryptInvocation {
                input: "m.enc".into(),
                output: "m.apr".into(),
                key_file: Some("k".into()),
                output_exists: true,
                ..Default::default()
            },
        ),
        (
            "in==out",
            DecryptInvocation {
                input: "x".into(),
                output: "x".into(),
                key_file: Some("k".into()),
                ..Default::default()
            },
        ),
    ];

    for (label, inv) in cases {
        println!("{label:>22}  →  {:?}", validate_invocation(inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy() -> DecryptInvocation {
        DecryptInvocation {
            input: "m.enc".into(),
            output: "m.apr".into(),
            key_file: Some("k".into()),
            ..Default::default()
        }
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_with_key_file_passes() {
        assert_eq!(validate_invocation(&happy()), InvocationVerdict::Ok);
    }

    #[test]
    fn stdin_passphrase_path_passes() {
        let inv = DecryptInvocation {
            input: "m.enc".into(),
            output: "m.apr".into(),
            passphrase_via_stdin: true,
            ..Default::default()
        };
        assert_eq!(validate_invocation(&inv), InvocationVerdict::Ok);
    }

    #[test]
    fn neither_key_nor_stdin_rejected() {
        let inv = DecryptInvocation {
            input: "m.enc".into(),
            output: "m.apr".into(),
            ..Default::default()
        };
        assert_eq!(validate_invocation(&inv), InvocationVerdict::NoKeySource);
    }

    #[test]
    fn input_equals_output_rejected() {
        let mut inv = happy();
        inv.input = "x".into();
        inv.output = "x".into();
        assert_eq!(validate_invocation(&inv), InvocationVerdict::InvalidPaths);
    }

    #[test]
    fn existing_output_without_force_rejected() {
        // Important: NEVER silently overwrite a decrypted artifact — operator
        // must opt in via --force so an old decryption isn't clobbered.
        let mut inv = happy();
        inv.output_exists = true;
        assert_eq!(
            validate_invocation(&inv),
            InvocationVerdict::OutputExistsWithoutForce
        );
    }

    #[test]
    fn existing_output_with_force_passes() {
        let mut inv = happy();
        inv.output_exists = true;
        inv.force = true;
        assert_eq!(validate_invocation(&inv), InvocationVerdict::Ok);
    }
}
