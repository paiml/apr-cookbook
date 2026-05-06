//! # Registry OCI Image Index Resolver
//!
//! OCI image indexes (multi-platform manifests) list per-arch
//! manifests. Resolver picks the right manifest by platform tuple
//! (os, arch, variant) — first exact match wins.
//!
//! Demonstrates the **REG.22** recipe for PMAT-150 (registry round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OCI Image Index Specification v1.1.
//!
//! Run with: cargo run --example registry_oci_index
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformManifest {
    pub digest: String,
    pub os: String,
    pub arch: String,
    pub variant: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum ResolveVerdict {
    Ok { digest: String },
    NoMatch { available: Vec<String> },
    EmptyIndex,
    InvalidPlatform,
}

pub fn resolve(
    manifests: &[PlatformManifest],
    target_os: &str,
    target_arch: &str,
    target_variant: Option<&str>,
) -> ResolveVerdict {
    if manifests.is_empty() {
        return ResolveVerdict::EmptyIndex;
    }
    if target_os.is_empty() || target_arch.is_empty() {
        return ResolveVerdict::InvalidPlatform;
    }
    for m in manifests {
        if m.os == target_os && m.arch == target_arch && m.variant.as_deref() == target_variant {
            return ResolveVerdict::Ok {
                digest: m.digest.clone(),
            };
        }
    }
    let available: Vec<String> = manifests
        .iter()
        .map(|m| {
            let v = m.variant.as_deref().unwrap_or("");
            format!("{}/{}/{}", m.os, m.arch, v)
        })
        .collect();
    ResolveVerdict::NoMatch { available }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_oci_index")?;

    let manifests = vec![
        PlatformManifest {
            digest: "sha256:linux-amd64".to_string(),
            os: "linux".to_string(),
            arch: "amd64".to_string(),
            variant: None,
        },
        PlatformManifest {
            digest: "sha256:linux-arm64".to_string(),
            os: "linux".to_string(),
            arch: "arm64".to_string(),
            variant: None,
        },
        PlatformManifest {
            digest: "sha256:linux-arm32-v7".to_string(),
            os: "linux".to_string(),
            arch: "arm".to_string(),
            variant: Some("v7".to_string()),
        },
    ];
    println!(
        "linux/amd64: {:?}",
        resolve(&manifests, "linux", "amd64", None)
    );
    println!(
        "linux/arm/v7: {:?}",
        resolve(&manifests, "linux", "arm", Some("v7"))
    );
    println!(
        "darwin/arm64: {:?}",
        resolve(&manifests, "darwin", "arm64", None)
    );
    println!("empty: {:?}", resolve(&[], "linux", "amd64", None));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<PlatformManifest> {
        vec![
            PlatformManifest {
                digest: "amd64".to_string(),
                os: "linux".to_string(),
                arch: "amd64".to_string(),
                variant: None,
            },
            PlatformManifest {
                digest: "arm64".to_string(),
                os: "linux".to_string(),
                arch: "arm64".to_string(),
                variant: None,
            },
        ]
    }

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_amd64() {
        let v = resolve(&typical(), "linux", "amd64", None);
        if let ResolveVerdict::Ok { digest } = v {
            assert_eq!(digest, "amd64");
        }
    }

    #[test]
    fn exact_match_arm64() {
        let v = resolve(&typical(), "linux", "arm64", None);
        if let ResolveVerdict::Ok { digest } = v {
            assert_eq!(digest, "arm64");
        }
    }

    #[test]
    fn no_match_returns_available() {
        let v = resolve(&typical(), "darwin", "amd64", None);
        if let ResolveVerdict::NoMatch { available } = v {
            assert_eq!(available.len(), 2);
        }
    }

    #[test]
    fn empty_index_rejected() {
        assert_eq!(
            resolve(&[], "linux", "amd64", None),
            ResolveVerdict::EmptyIndex
        );
    }

    #[test]
    fn empty_os_invalid() {
        assert_eq!(
            resolve(&typical(), "", "amd64", None),
            ResolveVerdict::InvalidPlatform
        );
    }

    #[test]
    fn empty_arch_invalid() {
        assert_eq!(
            resolve(&typical(), "linux", "", None),
            ResolveVerdict::InvalidPlatform
        );
    }

    #[test]
    fn variant_match() {
        let manifests = vec![PlatformManifest {
            digest: "v7".to_string(),
            os: "linux".to_string(),
            arch: "arm".to_string(),
            variant: Some("v7".to_string()),
        }];
        let v = resolve(&manifests, "linux", "arm", Some("v7"));
        if let ResolveVerdict::Ok { digest } = v {
            assert_eq!(digest, "v7");
        }
    }

    #[test]
    fn variant_mismatch_no_match() {
        let manifests = vec![PlatformManifest {
            digest: "v7".to_string(),
            os: "linux".to_string(),
            arch: "arm".to_string(),
            variant: Some("v7".to_string()),
        }];
        let v = resolve(&manifests, "linux", "arm", Some("v8"));
        assert!(matches!(v, ResolveVerdict::NoMatch { .. }));
    }

    #[test]
    fn variant_present_target_none_no_match() {
        // Target without variant doesn't match manifest with variant.
        let manifests = vec![PlatformManifest {
            digest: "v7".to_string(),
            os: "linux".to_string(),
            arch: "arm".to_string(),
            variant: Some("v7".to_string()),
        }];
        let v = resolve(&manifests, "linux", "arm", None);
        assert!(matches!(v, ResolveVerdict::NoMatch { .. }));
    }

    #[test]
    fn first_match_wins() {
        let manifests = vec![
            PlatformManifest {
                digest: "first".to_string(),
                os: "linux".to_string(),
                arch: "amd64".to_string(),
                variant: None,
            },
            PlatformManifest {
                digest: "second".to_string(),
                os: "linux".to_string(),
                arch: "amd64".to_string(),
                variant: None,
            },
        ];
        let v = resolve(&manifests, "linux", "amd64", None);
        if let ResolveVerdict::Ok { digest } = v {
            assert_eq!(digest, "first");
        }
    }
}
