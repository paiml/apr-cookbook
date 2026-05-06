//! # apr export --output — Filename Template Validator
//!
//! `apr export <FILE> --output <TEMPLATE>` accepts a filename template
//! with `{model}`, `{format}`, `{dtype}`, `{date}` placeholders. Rules:
//! at least one placeholder OR a literal filename; no path traversal
//! (`..`); extension matches format (e.g., GGUF → .gguf). This recipe
//! builds the validator + renderer.
//!
//! Demonstrates the **EXP.6** recipe for PMAT-117 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + filename safety conventions
//!
//! Run with: cargo run --example cli_export_output_naming_pattern
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NameVerdict {
    Ok(String),
    PathTraversal,
    EmptyTemplate,
    UnknownPlaceholder { placeholder: String },
    ExtensionMismatch { expected: String, got: String },
}

pub fn render(template: &str, model: &str, format: &str, dtype: &str, date: &str) -> NameVerdict {
    if template.is_empty() {
        return NameVerdict::EmptyTemplate;
    }
    if template.contains("..") || template.starts_with('/') {
        return NameVerdict::PathTraversal;
    }
    let mut out = template.to_string();
    let known = [
        ("{model}", model),
        ("{format}", format),
        ("{dtype}", dtype),
        ("{date}", date),
    ];
    for (k, v) in known {
        out = out.replace(k, v);
    }
    if let Some(start) = out.find('{') {
        if let Some(end) = out[start..].find('}') {
            let placeholder = &out[start..=start + end];
            return NameVerdict::UnknownPlaceholder {
                placeholder: placeholder.to_string(),
            };
        }
    }
    let expected_ext = format_to_ext(format);
    if !expected_ext.is_empty() && !out.ends_with(&format!(".{expected_ext}")) {
        let got = out.rsplit('.').next().unwrap_or("").to_string();
        return NameVerdict::ExtensionMismatch {
            expected: expected_ext.to_string(),
            got,
        };
    }
    NameVerdict::Ok(out)
}

fn format_to_ext(format: &str) -> &'static str {
    match format {
        "gguf" => "gguf",
        "onnx" => "onnx",
        "safetensors" => "safetensors",
        "apr" => "apr",
        _ => "",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_output_naming_pattern")?;

    let cases = [
        ("{model}-{dtype}.gguf", "llama-3", "gguf", "q4", "20260506"),
        ("../escape.gguf", "x", "gguf", "x", "x"),
        ("model.onnx", "x", "gguf", "x", "x"),
        ("{unknown}.gguf", "x", "gguf", "x", "x"),
    ];
    for (t, m, f, d, dt) in cases {
        println!("{t:>30} → {:?}", render(t, m, f, d, dt));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_template_renders() {
        let v = render("{model}-{dtype}.gguf", "llama-3", "gguf", "q4", "");
        assert_eq!(v, NameVerdict::Ok("llama-3-q4.gguf".into()));
    }

    #[test]
    fn empty_template_rejected() {
        assert_eq!(
            render("", "m", "gguf", "fp16", ""),
            NameVerdict::EmptyTemplate
        );
    }

    #[test]
    fn path_traversal_rejected() {
        let v = render("../bad.gguf", "m", "gguf", "fp16", "");
        assert_eq!(v, NameVerdict::PathTraversal);
    }

    #[test]
    fn absolute_path_rejected() {
        let v = render("/etc/passwd.gguf", "m", "gguf", "fp16", "");
        assert_eq!(v, NameVerdict::PathTraversal);
    }

    #[test]
    fn unknown_placeholder_rejected() {
        let v = render("{nope}.gguf", "m", "gguf", "fp16", "");
        assert!(matches!(v, NameVerdict::UnknownPlaceholder { .. }));
    }

    #[test]
    fn extension_mismatch_rejected() {
        // Format is gguf but template ends in .onnx.
        let v = render("model.onnx", "m", "gguf", "fp16", "");
        assert!(matches!(v, NameVerdict::ExtensionMismatch { .. }));
    }

    #[test]
    fn literal_filename_with_correct_extension_passes() {
        let v = render("model.gguf", "m", "gguf", "fp16", "");
        assert_eq!(v, NameVerdict::Ok("model.gguf".into()));
    }

    #[test]
    fn date_placeholder_substitutes() {
        let v = render("{model}-{date}.apr", "llama", "apr", "fp16", "20260506");
        assert_eq!(v, NameVerdict::Ok("llama-20260506.apr".into()));
    }

    #[test]
    fn unknown_format_skips_extension_check() {
        // Format "unknown" has no extension mapping → no mismatch error.
        let v = render("model.bin", "m", "unknown", "fp16", "");
        assert_eq!(v, NameVerdict::Ok("model.bin".into()));
    }
}
