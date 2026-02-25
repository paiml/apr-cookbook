//! # Import Model from HuggingFace
//!
//! **CLI equivalent:** `apr import hf://org/repo`
//!
//! Demonstrates the complete workflow for importing a model from
//! HuggingFace Hub into the APR v2 format. Parses HF URIs, simulates
//! downloading model weights and metadata, and converts everything
//! into a valid APR v2 bundle.
//!
//! ## Sections
//! 1. URI parsing — decompose `hf://org/repo@revision/file` into components
//! 2. Metadata extraction — simulate fetching model config and tokenizer info
//! 3. Weight conversion — transform downloaded bytes into APR tensor layout
//! 4. APR bundle creation — produce a valid APR v2 bundle with all metadata

use apr_cookbook::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Parsed HuggingFace model reference.
#[derive(Debug, Clone, PartialEq)]
struct HfModelRef {
    org: String,
    repo: String,
    revision: Option<String>,
    file: Option<String>,
}

/// Metadata fetched alongside model weights.
#[derive(Debug, Clone)]
struct HfMetadata {
    model_type: String,
    num_parameters: usize,
    hidden_size: usize,
    num_layers: usize,
    license: String,
    tags: Vec<String>,
}

/// Result of a simulated download.
struct DownloadResult {
    metadata: HfMetadata,
    weight_bytes: Vec<u8>,
    tensor_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// URI parsing
// ---------------------------------------------------------------------------

/// Parse a HuggingFace URI into its components.
///
/// Accepted formats:
/// - `hf://org/repo`
/// - `hf://org/repo@revision`
/// - `hf://org/repo@revision/file.safetensors`
/// - `hf://org/repo/file.safetensors`
fn parse_hf_uri(uri: &str) -> Result<HfModelRef> {
    let stripped = uri
        .strip_prefix("hf://")
        .ok_or_else(|| CookbookError::invalid_format("URI must start with hf://"))?;

    let parts: Vec<&str> = stripped.splitn(3, '/').collect();
    if parts.len() < 2 {
        return Err(CookbookError::invalid_format(
            "URI must contain at least org/repo",
        ));
    }

    let org = parts[0].to_string();

    // Handle revision syntax: repo@revision
    let (repo, revision) = if let Some(at_pos) = parts[1].find('@') {
        let repo_name = parts[1][..at_pos].to_string();
        let rev = parts[1][at_pos + 1..].to_string();
        (repo_name, Some(rev))
    } else {
        (parts[1].to_string(), None)
    };

    let file: Option<String> = parts.get(2).map(|f| (*f).to_string());

    if org.is_empty() || repo.is_empty() {
        return Err(CookbookError::invalid_format(
            "org and repo must be non-empty",
        ));
    }

    Ok(HfModelRef {
        org,
        repo,
        revision,
        file,
    })
}

// ---------------------------------------------------------------------------
// Simulated download
// ---------------------------------------------------------------------------

/// Simulate downloading a model from HuggingFace Hub.
///
/// In production this would use the HF API to fetch model files.
/// Here we generate deterministic synthetic weights based on the model ref.
fn simulate_download(model_ref: &HfModelRef) -> DownloadResult {
    let seed = hash_name_to_seed(&format!("{}/{}", model_ref.org, model_ref.repo));
    let hidden_size = 64 + (seed % 64) as usize;
    let num_layers = 2 + (seed % 6) as usize;
    let num_params = hidden_size * hidden_size * num_layers;

    let tensor_names: Vec<String> = (0..num_layers)
        .flat_map(|l| {
            vec![
                format!("layer.{l}.attention.weight"),
                format!("layer.{l}.ffn.weight"),
            ]
        })
        .collect();

    let weight_bytes = generate_model_payload(seed, num_params);

    let metadata = HfMetadata {
        model_type: "transformer".to_string(),
        num_parameters: num_params,
        hidden_size,
        num_layers,
        license: "apache-2.0".to_string(),
        tags: vec![
            "text-generation".to_string(),
            "rust".to_string(),
            "apr".to_string(),
        ],
    };

    DownloadResult {
        metadata,
        weight_bytes,
        tensor_names,
    }
}

// ---------------------------------------------------------------------------
// Conversion to APR v2
// ---------------------------------------------------------------------------

/// Convert downloaded weights and metadata into an APR v2 bundle.
fn convert_to_apr(download: &DownloadResult) -> Vec<u8> {
    let bytes_per_tensor = download.weight_bytes.len() / download.tensor_names.len().max(1);
    let dim = (bytes_per_tensor / 4).max(1); // f32 elements
    let side = (dim as f64).sqrt() as usize;

    let mut builder = ModelBundleV2::new()
        .with_name(&download.metadata.model_type)
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32);

    for (i, name) in download.tensor_names.iter().enumerate() {
        let start = i * bytes_per_tensor;
        let end = ((i + 1) * bytes_per_tensor).min(download.weight_bytes.len());
        let tensor_bytes = download.weight_bytes[start..end].to_vec();
        builder = builder.add_tensor(name, vec![side, side], tensor_bytes);
    }

    builder.build()
}

/// Build a metadata map for the APR bundle from HF metadata.
fn build_metadata_map(metadata: &HfMetadata) -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("model_type".to_string(), metadata.model_type.clone());
    map.insert(
        "num_parameters".to_string(),
        metadata.num_parameters.to_string(),
    );
    map.insert("hidden_size".to_string(), metadata.hidden_size.to_string());
    map.insert("num_layers".to_string(), metadata.num_layers.to_string());
    map.insert("license".to_string(), metadata.license.clone());
    map.insert("tags".to_string(), metadata.tags.join(","));
    map.insert("source".to_string(), "huggingface".to_string());
    map
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_import_hf")?;

    // Section 1: URI parsing
    let uri = "hf://microsoft/phi-2@main/model.safetensors";
    let model_ref = parse_hf_uri(uri)?;
    println!("=== HuggingFace Import ===");
    println!("URI:      {uri}");
    println!("Org:      {}", model_ref.org);
    println!("Repo:     {}", model_ref.repo);
    println!(
        "Revision: {}",
        model_ref.revision.as_deref().unwrap_or("latest")
    );
    println!("File:     {}", model_ref.file.as_deref().unwrap_or("(all)"));
    println!();

    // Section 2: Metadata extraction
    let download = simulate_download(&model_ref);
    println!("=== Downloaded Metadata ===");
    println!("Model type:     {}", download.metadata.model_type);
    println!("Parameters:     {}", download.metadata.num_parameters);
    println!("Hidden size:    {}", download.metadata.hidden_size);
    println!("Layers:         {}", download.metadata.num_layers);
    println!("License:        {}", download.metadata.license);
    println!("Tensors:        {}", download.tensor_names.len());
    println!("Weight bytes:   {}", download.weight_bytes.len());
    println!();

    // Section 3: Weight conversion to APR
    let apr_bundle = convert_to_apr(&download);
    println!("=== APR v2 Bundle ===");
    println!("Bundle size:    {} bytes", apr_bundle.len());
    println!(
        "Magic:          {:?}",
        std::str::from_utf8(&apr_bundle[0..4]).unwrap_or("???")
    );
    assert_eq!(&apr_bundle[0..4], b"APR2", "Valid APR2 magic bytes");
    println!();

    // Section 4: Metadata map
    let meta_map = build_metadata_map(&download.metadata);
    println!("=== Metadata Map ({} entries) ===", meta_map.len());
    let mut keys: Vec<_> = meta_map.keys().collect();
    keys.sort();
    for key in keys {
        println!("  {key}: {}", meta_map[key]);
    }

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_uri() {
        let r = parse_hf_uri("hf://openai/whisper").unwrap();
        assert_eq!(r.org, "openai");
        assert_eq!(r.repo, "whisper");
        assert!(r.revision.is_none());
        assert!(r.file.is_none());
    }

    #[test]
    fn test_parse_uri_with_revision() {
        let r = parse_hf_uri("hf://meta/llama@v2").unwrap();
        assert_eq!(r.org, "meta");
        assert_eq!(r.repo, "llama");
        assert_eq!(r.revision.as_deref(), Some("v2"));
        assert!(r.file.is_none());
    }

    #[test]
    fn test_parse_uri_with_file() {
        let r = parse_hf_uri("hf://org/repo/weights.safetensors").unwrap();
        assert_eq!(r.file.as_deref(), Some("weights.safetensors"));
    }

    #[test]
    fn test_parse_uri_with_revision_and_file() {
        let r = parse_hf_uri("hf://ms/phi@main/model.safetensors").unwrap();
        assert_eq!(r.org, "ms");
        assert_eq!(r.repo, "phi");
        assert_eq!(r.revision.as_deref(), Some("main"));
        assert_eq!(r.file.as_deref(), Some("model.safetensors"));
    }

    #[test]
    fn test_parse_uri_missing_prefix() {
        assert!(parse_hf_uri("https://huggingface.co/org/repo").is_err());
    }

    #[test]
    fn test_parse_uri_missing_repo() {
        assert!(parse_hf_uri("hf://orgonly").is_err());
    }

    #[test]
    fn test_parse_uri_empty_org() {
        assert!(parse_hf_uri("hf:///repo").is_err());
    }

    #[test]
    fn test_download_produces_weights() {
        let model_ref = parse_hf_uri("hf://test/model").unwrap();
        let dl = simulate_download(&model_ref);
        assert!(!dl.weight_bytes.is_empty());
        assert!(!dl.tensor_names.is_empty());
        assert!(dl.metadata.num_parameters > 0);
    }

    #[test]
    fn test_download_deterministic() {
        let model_ref = parse_hf_uri("hf://test/model").unwrap();
        let dl1 = simulate_download(&model_ref);
        let dl2 = simulate_download(&model_ref);
        assert_eq!(dl1.weight_bytes, dl2.weight_bytes);
        assert_eq!(dl1.tensor_names, dl2.tensor_names);
    }

    #[test]
    fn test_convert_produces_valid_apr() {
        let model_ref = parse_hf_uri("hf://test/model").unwrap();
        let dl = simulate_download(&model_ref);
        let bundle = convert_to_apr(&dl);
        assert_eq!(&bundle[0..4], b"APR2");
        assert!(bundle.len() > 4);
    }

    #[test]
    fn test_metadata_map_complete() {
        let meta = HfMetadata {
            model_type: "bert".to_string(),
            num_parameters: 1000,
            hidden_size: 128,
            num_layers: 4,
            license: "mit".to_string(),
            tags: vec!["nlp".to_string()],
        };
        let map = build_metadata_map(&meta);
        assert_eq!(map["model_type"], "bert");
        assert_eq!(map["num_parameters"], "1000");
        assert_eq!(map["source"], "huggingface");
        assert!(map.contains_key("license"));
        assert!(map.contains_key("tags"));
    }

    #[test]
    fn test_different_repos_produce_different_weights() {
        let dl1 = simulate_download(&parse_hf_uri("hf://a/model1").unwrap());
        let dl2 = simulate_download(&parse_hf_uri("hf://b/model2").unwrap());
        // Different seeds should (very likely) produce different sizes or content
        assert_ne!(dl1.metadata.hidden_size, dl2.metadata.hidden_size);
    }
}
