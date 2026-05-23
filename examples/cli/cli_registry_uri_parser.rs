//! # apr registry — Model URI Parser
//!
//! Registry URIs follow `<scheme>://<owner>/<name>:<tag>` where scheme
//! ∈ {hf, s3, file, registry}, tag defaults to `latest`. Owner +
//! name use `[A-Za-z0-9_.-]`. This recipe builds the parser.
//!
//! Demonstrates the **REG.4** recipe for PMAT-114 (apr registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender REG-001 + RFC 3986 §3
//!
//! Run with: cargo run --example cli_registry_uri_parser
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub enum Scheme {
    HuggingFace,
    S3,
    File,
    Registry,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModelUri {
    pub scheme: Scheme,
    pub owner: String,
    pub name: String,
    pub tag: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    MissingScheme,
    UnknownScheme,
    MissingOwner,
    MissingName,
    InvalidIdentifier,
}

pub fn parse(uri: &str) -> std::result::Result<ModelUri, ParseError> {
    let (scheme_str, rest) = uri.split_once("://").ok_or(ParseError::MissingScheme)?;
    let scheme = match scheme_str {
        "hf" => Scheme::HuggingFace,
        "s3" => Scheme::S3,
        "file" => Scheme::File,
        "registry" => Scheme::Registry,
        _ => return Err(ParseError::UnknownScheme),
    };
    let (path, tag) = match rest.rsplit_once(':') {
        Some((p, t)) if !t.is_empty() => (p, t.to_string()),
        Some((p, _)) => (p, "latest".to_string()),
        None => (rest, "latest".to_string()),
    };
    let (owner, name) = path.split_once('/').ok_or(ParseError::MissingOwner)?;
    if owner.is_empty() {
        return Err(ParseError::MissingOwner);
    }
    if name.is_empty() {
        return Err(ParseError::MissingName);
    }
    if !is_valid_ident(owner) || !is_valid_ident(name) {
        return Err(ParseError::InvalidIdentifier);
    }
    Ok(ModelUri {
        scheme,
        owner: owner.into(),
        name: name.into(),
        tag,
    })
}

fn is_valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_uri_parser")?;

    for u in [
        "hf://meta-llama/Llama-3-8B:v1.0",
        "s3://my-bucket/model",
        "file://local/path:dev",
        "://no-scheme",
        "ftp://wrong/scheme",
        "hf://owner/",
    ] {
        println!("{u:<40}  →  {:?}", parse(u));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_hf_uri_parses() {
        let m = parse("hf://meta-llama/Llama-3-8B:v1.0").unwrap();
        assert_eq!(m.scheme, Scheme::HuggingFace);
        assert_eq!(m.owner, "meta-llama");
        assert_eq!(m.name, "Llama-3-8B");
        assert_eq!(m.tag, "v1.0");
    }

    #[test]
    fn missing_tag_defaults_to_latest() {
        let m = parse("hf://owner/model").unwrap();
        assert_eq!(m.tag, "latest");
    }

    #[test]
    fn empty_tag_after_colon_uses_default() {
        let m = parse("hf://owner/model:").unwrap();
        assert_eq!(m.tag, "latest");
    }

    #[test]
    fn missing_scheme_separator_rejected() {
        assert_eq!(parse("hf:owner/model"), Err(ParseError::MissingScheme));
    }

    #[test]
    fn unknown_scheme_rejected() {
        assert_eq!(parse("ftp://owner/model"), Err(ParseError::UnknownScheme));
    }

    #[test]
    fn missing_owner_or_name_rejected() {
        assert_eq!(parse("hf:///model"), Err(ParseError::MissingOwner));
        assert_eq!(parse("hf://owner/"), Err(ParseError::MissingName));
    }

    #[test]
    fn invalid_identifier_rejected() {
        // Spaces are not allowed.
        assert_eq!(
            parse("hf://owner name/model"),
            Err(ParseError::InvalidIdentifier)
        );
    }

    #[test]
    fn all_known_schemes_parse() {
        for (s, kind) in [
            ("hf", Scheme::HuggingFace),
            ("s3", Scheme::S3),
            ("file", Scheme::File),
            ("registry", Scheme::Registry),
        ] {
            let uri = format!("{s}://owner/model");
            let m = parse(&uri).unwrap();
            assert_eq!(m.scheme, kind);
        }
    }

    #[test]
    fn dots_dashes_underscores_in_names_allowed() {
        let m = parse("hf://my.org_team/model-name.v2").unwrap();
        assert_eq!(m.owner, "my.org_team");
        assert_eq!(m.name, "model-name.v2");
    }
}
