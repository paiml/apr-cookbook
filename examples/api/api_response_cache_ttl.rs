//! # API Response Cache TTL Picker
//!
//! Cacheable responses get a TTL based on resource type + freshness
//! requirements. Static models (rarely change): 24 h; metadata
//! (registry list): 5 min; inference results (deterministic with
//! seed): 1 h; user-specific (auth-bound): 0 (no-cache). This recipe
//! builds the picker + Cache-Control header validator.
//!
//! Demonstrates the **API.7** recipe for PMAT-125 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 9111 (HTTP Caching).
//!
//! Run with: cargo run --example api_response_cache_ttl
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    StaticModel,
    Metadata,
    DeterministicInference,
    UserSpecific,
    Streaming,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CacheVerdict {
    Cacheable { ttl_secs: u32 },
    NoCache,
}

pub fn pick_ttl(kind: ResourceKind, has_auth: bool) -> CacheVerdict {
    if has_auth && !matches!(kind, ResourceKind::StaticModel) {
        return CacheVerdict::NoCache;
    }
    match kind {
        ResourceKind::StaticModel => CacheVerdict::Cacheable { ttl_secs: 86_400 },
        ResourceKind::Metadata => CacheVerdict::Cacheable { ttl_secs: 300 },
        ResourceKind::DeterministicInference => CacheVerdict::Cacheable { ttl_secs: 3600 },
        ResourceKind::UserSpecific | ResourceKind::Streaming => CacheVerdict::NoCache,
    }
}

pub fn cache_control_header(verdict: &CacheVerdict) -> String {
    match verdict {
        CacheVerdict::NoCache => "no-store".into(),
        CacheVerdict::Cacheable { ttl_secs } => format!("public, max-age={ttl_secs}"),
    }
}

#[derive(Debug, PartialEq)]
pub enum HeaderVerdict {
    Ok {
        is_cacheable: bool,
        max_age: Option<u32>,
    },
    InvalidDirective,
}

pub fn parse_cache_control(header: &str) -> HeaderVerdict {
    let lower = header.to_ascii_lowercase();
    if lower.contains("no-store") || lower.contains("no-cache") || lower.contains("private") {
        return HeaderVerdict::Ok {
            is_cacheable: false,
            max_age: None,
        };
    }
    if let Some(rest) = lower.split_once("max-age=") {
        let value = rest.1.trim().split(',').next().unwrap_or("").trim();
        if let Ok(secs) = value.parse::<u32>() {
            return HeaderVerdict::Ok {
                is_cacheable: true,
                max_age: Some(secs),
            };
        }
        return HeaderVerdict::InvalidDirective;
    }
    if lower.contains("public") {
        HeaderVerdict::Ok {
            is_cacheable: true,
            max_age: None,
        }
    } else {
        HeaderVerdict::Ok {
            is_cacheable: false,
            max_age: None,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_response_cache_ttl")?;

    for k in [
        ResourceKind::StaticModel,
        ResourceKind::Metadata,
        ResourceKind::DeterministicInference,
        ResourceKind::UserSpecific,
        ResourceKind::Streaming,
    ] {
        let v = pick_ttl(k, false);
        println!("{k:?}  →  {v:?}  hdr={}", cache_control_header(&v));
    }
    println!(
        "auth StaticModel: {:?}",
        pick_ttl(ResourceKind::StaticModel, true)
    );
    println!(
        "auth Metadata: {:?}",
        pick_ttl(ResourceKind::Metadata, true)
    );

    for h in ["no-store", "public, max-age=300", "private", "max-age=foo"] {
        println!("{h} → {:?}", parse_cache_control(h));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn static_model_24h() {
        assert_eq!(
            pick_ttl(ResourceKind::StaticModel, false),
            CacheVerdict::Cacheable { ttl_secs: 86_400 }
        );
    }

    #[test]
    fn metadata_5min() {
        assert_eq!(
            pick_ttl(ResourceKind::Metadata, false),
            CacheVerdict::Cacheable { ttl_secs: 300 }
        );
    }

    #[test]
    fn deterministic_inference_1h() {
        assert_eq!(
            pick_ttl(ResourceKind::DeterministicInference, false),
            CacheVerdict::Cacheable { ttl_secs: 3600 }
        );
    }

    #[test]
    fn user_specific_no_cache() {
        assert_eq!(
            pick_ttl(ResourceKind::UserSpecific, false),
            CacheVerdict::NoCache
        );
    }

    #[test]
    fn streaming_no_cache() {
        assert_eq!(
            pick_ttl(ResourceKind::Streaming, false),
            CacheVerdict::NoCache
        );
    }

    #[test]
    fn auth_disables_cache_except_static() {
        assert_eq!(
            pick_ttl(ResourceKind::Metadata, true),
            CacheVerdict::NoCache
        );
        // Static model is cacheable even with auth (it's the model file itself).
        assert!(matches!(
            pick_ttl(ResourceKind::StaticModel, true),
            CacheVerdict::Cacheable { .. }
        ));
    }

    #[test]
    fn header_includes_max_age() {
        let v = CacheVerdict::Cacheable { ttl_secs: 60 };
        assert_eq!(cache_control_header(&v), "public, max-age=60");
    }

    #[test]
    fn no_cache_header_is_no_store() {
        assert_eq!(cache_control_header(&CacheVerdict::NoCache), "no-store");
    }

    #[test]
    fn parse_no_store_not_cacheable() {
        let v = parse_cache_control("no-store");
        assert!(matches!(
            v,
            HeaderVerdict::Ok {
                is_cacheable: false,
                ..
            }
        ));
    }

    #[test]
    fn parse_max_age_extracts_value() {
        let v = parse_cache_control("public, max-age=300");
        assert!(matches!(
            v,
            HeaderVerdict::Ok {
                is_cacheable: true,
                max_age: Some(300)
            }
        ));
    }

    #[test]
    fn parse_invalid_max_age_rejected() {
        let v = parse_cache_control("max-age=foo");
        assert_eq!(v, HeaderVerdict::InvalidDirective);
    }
}
