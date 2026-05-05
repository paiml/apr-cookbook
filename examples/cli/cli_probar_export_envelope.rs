//! # apr probar — Export Envelope (Format Selection)
//!
//! `apr probar <FILE> --format {json,png,both} -o ./probar-export` builds
//! the test-artifact bundle that `probar` consumes for visual regression.
//! This recipe models the export envelope (paths + content-type triples)
//! as a pure function so a CI pipeline can preview which artifacts will
//! land in the output directory before invoking the binary.
//!
//! Demonstrates the **PROBAR.3** recipe for PMAT-093 (apr probar coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-481 + probar visual regression suite
//!
//! Run with: cargo run --example cli_probar_export_envelope
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbarFormat {
    Json,
    Png,
    Both,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportArtifact {
    pub rel_path: PathBuf,
    pub content_type: &'static str,
}

pub fn build_export_envelope(
    output_dir: &str,
    layer_filter: Option<&str>,
    format: ProbarFormat,
    layer_names: &[&str],
) -> Vec<ExportArtifact> {
    let dir = PathBuf::from(output_dir);
    let mut out = Vec::new();
    let want_json = matches!(format, ProbarFormat::Json | ProbarFormat::Both);
    let want_png = matches!(format, ProbarFormat::Png | ProbarFormat::Both);

    for &layer in layer_names {
        if let Some(pat) = layer_filter {
            if !layer.contains(pat) {
                continue;
            }
        }
        let safe = layer.replace('.', "_");
        if want_json {
            out.push(ExportArtifact {
                rel_path: dir.join(format!("{safe}.json")),
                content_type: "application/json",
            });
        }
        if want_png {
            out.push(ExportArtifact {
                rel_path: dir.join(format!("{safe}.png")),
                content_type: "image/png",
            });
        }
    }
    // Always emit the manifest as the last artifact.
    out.push(ExportArtifact {
        rel_path: dir.join("manifest.json"),
        content_type: "application/json",
    });
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_probar_export_envelope")?;

    let layers = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.embed_tokens.weight",
    ];

    println!("=== both formats, no filter ===");
    for a in build_export_envelope("./probar-export", None, ProbarFormat::Both, &layers) {
        println!("  {}  [{}]", a.rel_path.display(), a.content_type);
    }

    println!("\n=== JSON only, layer 0 filter ===");
    for a in build_export_envelope("./out", Some("layers.0"), ProbarFormat::Json, &layers) {
        println!("  {}  [{}]", a.rel_path.display(), a.content_type);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_layers() -> Vec<&'static str> {
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ]
    }

    #[test]
    fn export_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn both_format_emits_json_and_png_per_layer() {
        let layers = sample_layers();
        let env = build_export_envelope("./d", None, ProbarFormat::Both, &layers);
        // 3 layers × (json + png) + 1 manifest = 7 artifacts.
        assert_eq!(env.len(), 7);
        assert!(env.iter().any(|a| a.content_type == "application/json"));
        assert!(env.iter().any(|a| a.content_type == "image/png"));
    }

    #[test]
    fn manifest_is_always_last_artifact() {
        // CI tooling assumes manifest.json is the last write — pinning order.
        let env = build_export_envelope("./d", None, ProbarFormat::Json, &sample_layers());
        let last = env.last().unwrap();
        assert!(last.rel_path.ends_with("manifest.json"));
    }

    #[test]
    fn json_only_format_skips_png() {
        let env = build_export_envelope("./d", None, ProbarFormat::Json, &sample_layers());
        assert!(env.iter().all(|a| a.content_type != "image/png"));
    }

    #[test]
    fn layer_filter_substring_match() {
        let env = build_export_envelope(
            "./d",
            Some("layers.0"),
            ProbarFormat::Json,
            &sample_layers(),
        );
        // 2 layer-0 artifacts (q_proj + gate_proj) + 1 manifest
        assert_eq!(env.len(), 3);
    }

    #[test]
    fn layer_filter_no_match_yields_only_manifest() {
        // A pattern matching nothing is allowed — manifest still emitted so
        // CI doesn't blow up on "no artifacts" (it sees an empty manifest).
        let env = build_export_envelope(
            "./d",
            Some("nonexistent"),
            ProbarFormat::Both,
            &sample_layers(),
        );
        assert_eq!(env.len(), 1);
        assert!(env[0].rel_path.ends_with("manifest.json"));
    }

    #[test]
    fn dot_in_layer_name_replaced_in_path() {
        // "model.layers.0.self_attn.q_proj.weight" → "model_layers_0_..."
        let env = build_export_envelope("./d", None, ProbarFormat::Json, &["a.b.c"]);
        let p = env[0].rel_path.to_string_lossy().to_string();
        assert!(p.contains("a_b_c.json"));
        assert!(!p.contains("a.b.c.json"));
    }
}
