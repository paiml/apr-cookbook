//! # Advanced Capability Manifest Validator
//!
//! Cookbook capability manifest declares: surface count, recipes-per-
//! surface min/max, total recipes, F-invariant compliance. This recipe
//! validates a manifest against actual counts + flags drift between
//! declared and actual.
//!
//! Demonstrates the **ADV.5** recipe for PMAT-128 (advanced coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SPEC-COVERAGE-001.
//!
//! Run with: cargo run --example adv_capability_manifest_validator
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

const F_INVARIANT_FLOOR: u32 = 3;

#[derive(Debug, Clone)]
pub struct Manifest {
    pub total_recipes: u32,
    pub surfaces: BTreeMap<String, u32>,
}

#[derive(Debug, PartialEq)]
pub enum ManifestVerdict {
    Ok,
    TotalMismatch { declared: u32, actual: u32 },
    FInvariantViolated { surface: String, count: u32 },
    EmptyManifest,
}

pub fn validate(manifest: &Manifest) -> ManifestVerdict {
    if manifest.surfaces.is_empty() {
        return ManifestVerdict::EmptyManifest;
    }
    let actual_total: u32 = manifest.surfaces.values().sum();
    if actual_total != manifest.total_recipes {
        return ManifestVerdict::TotalMismatch {
            declared: manifest.total_recipes,
            actual: actual_total,
        };
    }
    for (surface, count) in &manifest.surfaces {
        if *count < F_INVARIANT_FLOOR {
            return ManifestVerdict::FInvariantViolated {
                surface: surface.clone(),
                count: *count,
            };
        }
    }
    ManifestVerdict::Ok
}

#[derive(Debug, PartialEq)]
pub struct DriftReport {
    pub added_surfaces: Vec<String>,
    pub removed_surfaces: Vec<String>,
    pub count_changes: Vec<(String, i32)>,
}

pub fn compute_drift(prev: &Manifest, curr: &Manifest) -> DriftReport {
    let prev_keys: std::collections::BTreeSet<_> = prev.surfaces.keys().collect();
    let curr_keys: std::collections::BTreeSet<_> = curr.surfaces.keys().collect();
    let added: Vec<String> = curr_keys
        .difference(&prev_keys)
        .map(|s| (*s).clone())
        .collect();
    let removed: Vec<String> = prev_keys
        .difference(&curr_keys)
        .map(|s| (*s).clone())
        .collect();
    let mut changes = Vec::new();
    for surf in prev_keys.intersection(&curr_keys) {
        let p = prev.surfaces[surf.as_str()] as i32;
        let c = curr.surfaces[surf.as_str()] as i32;
        if p != c {
            changes.push(((*surf).clone(), c - p));
        }
    }
    DriftReport {
        added_surfaces: added,
        removed_surfaces: removed,
        count_changes: changes,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_capability_manifest_validator")?;

    let mut surfaces = BTreeMap::new();
    surfaces.insert("creation".into(), 9u32);
    surfaces.insert("bundling".into(), 13);
    surfaces.insert("conversion".into(), 9);
    let m = Manifest {
        total_recipes: 31,
        surfaces,
    };
    println!("validate: {:?}", validate(&m));

    let mut bad = m.clone();
    bad.surfaces.insert("under-floor".into(), 1);
    bad.total_recipes = 32;
    println!("under-floor: {:?}", validate(&bad));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        let mut surfaces = BTreeMap::new();
        surfaces.insert("a".into(), 3u32);
        surfaces.insert("b".into(), 5);
        surfaces.insert("c".into(), 7);
        Manifest {
            total_recipes: 15,
            surfaces,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn well_formed_manifest_passes() {
        assert_eq!(validate(&sample_manifest()), ManifestVerdict::Ok);
    }

    #[test]
    fn empty_rejected() {
        let m = Manifest {
            total_recipes: 0,
            surfaces: BTreeMap::new(),
        };
        assert_eq!(validate(&m), ManifestVerdict::EmptyManifest);
    }

    #[test]
    fn total_mismatch_detected() {
        let mut m = sample_manifest();
        m.total_recipes = 99;
        let v = validate(&m);
        assert!(matches!(v, ManifestVerdict::TotalMismatch { .. }));
    }

    #[test]
    fn f_invariant_violation_detected() {
        let mut m = sample_manifest();
        m.surfaces.insert("under-floor".into(), 1);
        m.total_recipes = 16;
        let v = validate(&m);
        assert!(matches!(
            v,
            ManifestVerdict::FInvariantViolated { count: 1, .. }
        ));
    }

    #[test]
    fn boundary_at_3_recipes_passes() {
        let mut m = BTreeMap::new();
        m.insert("a".into(), 3u32);
        let manifest = Manifest {
            total_recipes: 3,
            surfaces: m,
        };
        assert_eq!(validate(&manifest), ManifestVerdict::Ok);
    }

    #[test]
    fn drift_added_surface() {
        let prev = sample_manifest();
        let mut curr = prev.clone();
        curr.surfaces.insert("new".into(), 4);
        curr.total_recipes = 19;
        let d = compute_drift(&prev, &curr);
        assert_eq!(d.added_surfaces, vec!["new"]);
        assert!(d.removed_surfaces.is_empty());
    }

    #[test]
    fn drift_count_change() {
        let prev = sample_manifest();
        let mut curr = prev.clone();
        curr.surfaces.insert("b".into(), 8);
        curr.total_recipes = 18;
        let d = compute_drift(&prev, &curr);
        assert_eq!(d.count_changes, vec![("b".into(), 3)]);
    }

    #[test]
    fn drift_removed_surface() {
        let prev = sample_manifest();
        let mut curr = prev.clone();
        curr.surfaces.remove("c");
        curr.total_recipes = 8;
        let d = compute_drift(&prev, &curr);
        assert_eq!(d.removed_surfaces, vec!["c"]);
    }

    #[test]
    fn no_drift_when_identical() {
        let prev = sample_manifest();
        let curr = sample_manifest();
        let d = compute_drift(&prev, &curr);
        assert!(d.added_surfaces.is_empty());
        assert!(d.removed_surfaces.is_empty());
        assert!(d.count_changes.is_empty());
    }
}
