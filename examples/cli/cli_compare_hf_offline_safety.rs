//! # apr compare-hf — `--offline` Network Safety
//!
//! `apr compare-hf --offline <APR> <HF>` guarantees zero network egress: the
//! Hugging Face checkpoint must already be on disk under `HF_HOME` (or the
//! supplied path), and any attempted hub fetch is treated as a hard error
//! rather than a silent download. This recipe exercises the resolver
//! decision tree that decides "local cache / explicit path / refuse".
//!
//! Demonstrates the **CMPHF.3** recipe for PMAT-088 (apr compare-hf coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CMPHF-003 + IIUR offline guarantee
//!
//! Run with: cargo run --example cli_compare_hf_offline_safety
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
enum HfSource {
    LocalPath(PathBuf),
    HfCacheHit(PathBuf),
    HubFetchRequired,
}

struct OfflineResolver<'a> {
    hf_home: &'a Path,
    cache_index: &'a [&'a str], // model ids known to be on-disk
    offline: bool,
}

impl OfflineResolver<'_> {
    fn resolve(&self, requested: &str) -> Result<HfSource> {
        if Path::new(requested).is_absolute() || requested.starts_with("./") {
            return Ok(HfSource::LocalPath(PathBuf::from(requested)));
        }
        if self.cache_index.contains(&requested) {
            // Mirror HF cache layout: $HF_HOME/hub/models--<org>--<name>
            let mangled = requested.replace('/', "--");
            let path = self.hf_home.join("hub").join(format!("models--{mangled}"));
            return Ok(HfSource::HfCacheHit(path));
        }
        if self.offline {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "model {requested:?} not in local HF cache and --offline forbids hub fetch"
            )));
        }
        Ok(HfSource::HubFetchRequired)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compare_hf_offline_safety")?;

    let resolver = OfflineResolver {
        hf_home: Path::new("/tmp/fake-hf-home"),
        cache_index: &["Qwen/Qwen2.5-Coder-7B-Instruct"],
        offline: true,
    };

    // Hit, explicit path, miss-with-refusal
    let cases = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "/abs/path/to/local.safetensors",
        "meta-llama/Llama-3.1-8B-Instruct",
    ];
    for req in cases {
        match resolver.resolve(req) {
            Ok(src) => println!("{req:>50}  →  {src:?}"),
            Err(e) => println!("{req:>50}  →  REFUSED ({e})"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn cache_hit_does_not_fetch() {
        let r = OfflineResolver {
            hf_home: Path::new("/tmp/hf"),
            cache_index: &["org/model"],
            offline: true,
        };
        let resolved = r.resolve("org/model").unwrap();
        assert!(matches!(resolved, HfSource::HfCacheHit(_)));
    }

    #[test]
    fn explicit_local_path_bypasses_cache() {
        let r = OfflineResolver {
            hf_home: Path::new("/tmp/hf"),
            cache_index: &[],
            offline: true,
        };
        let resolved = r.resolve("/data/local.safetensors").unwrap();
        assert_eq!(
            resolved,
            HfSource::LocalPath(PathBuf::from("/data/local.safetensors"))
        );
    }

    #[test]
    fn offline_miss_is_hard_error() {
        // The whole point of --offline: silent hub fetch is a contract
        // violation, not a fallback.
        let r = OfflineResolver {
            hf_home: Path::new("/tmp/hf"),
            cache_index: &[],
            offline: true,
        };
        let err = r.resolve("org/uncached-model");
        assert!(err.is_err());
    }

    #[test]
    fn online_miss_falls_through_to_hub() {
        // When --offline is NOT set, missing models trigger a fetch (the
        // legacy behavior). This recipe is here so the offline branch
        // can be tested in isolation.
        let r = OfflineResolver {
            hf_home: Path::new("/tmp/hf"),
            cache_index: &[],
            offline: false,
        };
        let resolved = r.resolve("org/uncached-model").unwrap();
        assert_eq!(resolved, HfSource::HubFetchRequired);
    }
}
