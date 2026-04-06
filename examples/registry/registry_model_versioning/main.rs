#![allow(unused_imports)]
//! # Recipe: Model Version Management
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Semantic versioning, compatibility checking, dependency resolution,
//! upgrade paths, and lifecycle management for model registries.
//!
//! ## QA: Build, test, clippy, fmt PASS. Proptests (50+ cases).
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_model_versioning")?;
    println!("=== Recipe: {} ===\n", ctx.name());
    let mut registry = register_model_versions(&mut ctx);
    check_compatibility_matrix(&mut registry, &mut ctx);
    resolve_dependencies(&mut registry, &mut ctx);
    compute_upgrade_paths(&registry);
    apply_lifecycle_transitions(&mut registry);
    print_version_history(&registry, &mut ctx);
    println!("=== Recipe complete ===");
    Ok(())
}

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    fn v(a: u32, b: u32, c: u32) -> SemanticVersion {
        SemanticVersion::new(a, b, c)
    }
    fn mv(
        name: &str,
        ver: SemanticVersion,
        status: VersionStatus,
        parent: Option<SemanticVersion>,
        changes: &str,
    ) -> ModelVersion {
        ModelVersion {
            name: name.into(),
            version: ver,
            status,
            parent_version: parent,
            changes: changes.into(),
        }
    }

    #[test]
    fn test_version_parse_display_ordering() {
        assert_eq!(SemanticVersion::parse("3.2.1"), Some(v(3, 2, 1)));
        assert_eq!(SemanticVersion::parse("1.2"), None);
        assert_eq!(format!("{}", v(2, 10, 5)), "2.10.5");
        assert!(v(1, 0, 0) < v(1, 1, 0) && v(1, 1, 0) < v(2, 0, 0));
        assert!(v(2, 0, 0).is_breaking_from(&v(1, 9, 9)));
        assert!(!v(1, 1, 0).is_breaking_from(&v(1, 0, 0)));
    }
    #[test]
    fn test_status_transitions() {
        assert!(VersionStatus::Alpha.can_transition_to(&VersionStatus::Beta));
        assert!(!VersionStatus::Alpha.can_transition_to(&VersionStatus::Stable));
        assert!(VersionStatus::Beta.can_transition_to(&VersionStatus::Stable));
        assert!(!VersionStatus::Archived.can_transition_to(&VersionStatus::Stable));
        assert!(VersionStatus::Deprecated.can_transition_to(&VersionStatus::Archived));
    }
    #[test]
    fn test_version_range() {
        let r = VersionRange::new(v(1, 0, 0), v(2, 0, 0));
        assert!(r.contains(&v(1, 0, 0)) && r.contains(&v(1, 5, 0)) && r.contains(&v(2, 0, 0)));
        assert!(!r.contains(&v(0, 9, 0)) && !r.contains(&v(2, 0, 1)));
    }
    #[test]
    fn test_registry_sort_and_compat() {
        let mut reg = VersionRegistry::new();
        reg.register(mv("t", v(2, 0, 0), VersionStatus::Stable, None, "v2"));
        reg.register(mv("t", v(1, 0, 0), VersionStatus::Stable, None, "v1"));
        assert_eq!(reg.versions["t"][0].version, v(1, 0, 0));
        reg.add_compatibility_rule(CompatibilityRule {
            source_model: "A".into(),
            source_range: VersionRange::new(v(1, 0, 0), v(1, 9, 0)),
            target_model: "B".into(),
            target_range: VersionRange::new(v(2, 0, 0), v(2, 5, 0)),
            compatible: true,
        });
        assert!(reg.check_compatibility("A", &v(1, 3, 0), "B", &v(2, 1, 0)));
        assert!(!reg.check_compatibility("A", &v(1, 0, 0), "B", &v(1, 0, 0)));
    }
    #[test]
    fn test_dependency_resolution() {
        let mut reg = VersionRegistry::new();
        reg.register(mv("enc", v(2, 0, 0), VersionStatus::Stable, None, "r"));
        reg.register(mv("enc", v(2, 1, 0), VersionStatus::Stable, None, "u"));
        reg.add_dependency(
            "det",
            &v(3, 0, 0),
            "enc",
            VersionRange::new(v(2, 0, 0), v(2, 1, 0)),
        );
        let res = reg.resolve_dependencies("det", &v(3, 0, 0));
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, v(2, 1, 0));
        assert!(reg.resolve_dependencies("m", &v(1, 0, 0)).is_empty());
    }
    #[test]
    fn test_upgrade_path() {
        let mut reg = VersionRegistry::new();
        for (maj, min) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
            let parent = if maj == 1 && min == 0 {
                None
            } else {
                Some(v(
                    if min == 0 { maj - 1 } else { maj },
                    if min == 0 { 1 } else { min - 1 },
                    0,
                ))
            };
            reg.register(mv(
                "m",
                v(maj, min, 0),
                VersionStatus::Stable,
                parent,
                &format!("v{maj}.{min}"),
            ));
        }
        let path = reg
            .find_upgrade_path("m", &v(1, 0, 0), &v(2, 1, 0))
            .expect("path");
        assert_eq!(path.steps.len(), 4);
        assert_eq!(path.breaking_changes, 1);
        assert!(reg
            .find_upgrade_path("x", &v(1, 0, 0), &v(2, 0, 0))
            .is_none());
    }
    #[test]
    fn test_transition_and_diff() {
        let mut reg = VersionRegistry::new();
        reg.register(mv("m", v(1, 0, 0), VersionStatus::Alpha, None, "init"));
        assert!(reg
            .transition_status("m", &v(1, 0, 0), VersionStatus::Beta)
            .is_ok());
        assert_eq!(reg.versions["m"][0].status, VersionStatus::Beta);
        let mut reg2 = VersionRegistry::new();
        reg2.register(mv("m", v(1, 0, 0), VersionStatus::Archived, None, "i"));
        assert!(reg2
            .transition_status("m", &v(1, 0, 0), VersionStatus::Stable)
            .is_err());
        let mut reg3 = VersionRegistry::new();
        reg3.register(mv("m", v(1, 0, 0), VersionStatus::Archived, None, "i"));
        reg3.register(mv(
            "m",
            v(1, 1, 0),
            VersionStatus::Deprecated,
            Some(v(1, 0, 0)),
            "p",
        ));
        reg3.register(mv(
            "m",
            v(2, 0, 0),
            VersionStatus::Stable,
            Some(v(1, 1, 0)),
            "b",
        ));
        let d = reg3.version_diff("m", &v(1, 0, 0), &v(2, 0, 0));
        assert_eq!(d.intermediate_count, 1);
        assert_eq!(d.breaking_count, 1);
    }
    #[test]
    fn test_main_runs() {
        assert!(main().is_ok());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test]
        fn prop_version_ordering_total(a0 in 0u32..10, a1 in 0u32..20, a2 in 0u32..30, b0 in 0u32..10, b1 in 0u32..20, b2 in 0u32..30) {
            let (a, b) = (SemanticVersion::new(a0,a1,a2), SemanticVersion::new(b0,b1,b2));
            prop_assert_eq!((a<b) as u8 + (a==b) as u8 + (a>b) as u8, 1);
        }
        #[test]
        fn prop_parse_roundtrip(maj in 0u32..100, min in 0u32..100, pat in 0u32..100) {
            let v = SemanticVersion::new(maj, min, pat);
            prop_assert_eq!(SemanticVersion::parse(&format!("{v}")), Some(v));
        }
        #[test]
        fn prop_register_sorted(n in 1usize..15) {
            let mut reg = VersionRegistry::new();
            for i in (0..n).rev() { reg.register(ModelVersion { name: "m".into(), version: SemanticVersion::new(1, i as u32, 0), status: VersionStatus::Stable, parent_version: None, changes: "t".into() }); }
            let vs = &reg.versions["m"];
            for w in vs.windows(2) { prop_assert!(w[0].version <= w[1].version); }
        }
    }
}
