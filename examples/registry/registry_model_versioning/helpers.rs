#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

pub fn register_model_versions(ctx: &mut RecipeContext) -> VersionRegistry {
    println!("--- Section 1: Register Model Versions ---");
    let parsed = SemanticVersion::parse("2.1.0");
    println!(
        "  Parsed \"2.1.0\" => {}",
        parsed
            .as_ref()
            .map_or("invalid".to_string(), |v| format!("{v}"))
    );
    let mut registry = VersionRegistry::new();
    let fd = "fraud-detector";
    let fe = "feature-encoder";
    #[allow(clippy::type_complexity)]
    let versions: &[(
        &str,
        u32,
        u32,
        u32,
        VersionStatus,
        Option<(u32, u32, u32)>,
        &str,
    )] = &[
        (
            fd,
            1,
            0,
            0,
            VersionStatus::Archived,
            None,
            "Initial release",
        ),
        (
            fd,
            1,
            1,
            0,
            VersionStatus::Deprecated,
            Some((1, 0, 0)),
            "Feature engineering",
        ),
        (
            fd,
            1,
            2,
            0,
            VersionStatus::Deprecated,
            Some((1, 1, 0)),
            "Threshold tuning",
        ),
        (
            fd,
            2,
            0,
            0,
            VersionStatus::Stable,
            Some((1, 2, 0)),
            "Breaking: embeddings",
        ),
        (
            fd,
            2,
            1,
            0,
            VersionStatus::Stable,
            Some((2, 0, 0)),
            "Real-time scoring",
        ),
        (
            fd,
            3,
            0,
            0,
            VersionStatus::Beta,
            Some((2, 1, 0)),
            "Breaking: transformer",
        ),
        (
            fd,
            3,
            1,
            0,
            VersionStatus::Alpha,
            Some((3, 0, 0)),
            "Multi-modal inputs",
        ),
        (
            fe,
            1,
            0,
            0,
            VersionStatus::Deprecated,
            None,
            "One-hot encoding",
        ),
        (
            fe,
            2,
            0,
            0,
            VersionStatus::Stable,
            Some((1, 0, 0)),
            "Breaking: embeddings",
        ),
        (
            fe,
            2,
            1,
            0,
            VersionStatus::Stable,
            Some((2, 0, 0)),
            "Caching layer",
        ),
    ];
    for &(name, maj, min, pat, ref status, parent, changes) in versions {
        registry.register(ModelVersion {
            name: name.to_string(),
            version: SemanticVersion::new(maj, min, pat),
            status: status.clone(),
            parent_version: parent.map(|(a, b, c)| SemanticVersion::new(a, b, c)),
            changes: changes.to_string(),
        });
    }
    ctx.record_metric(
        "total_versions",
        registry.versions.values().map(Vec::len).sum::<usize>() as i64,
    );
    for (mn, mvs) in &registry.versions {
        println!("  {} ({} versions):", mn, mvs.len());
        for mv in mvs {
            println!("    v{} [{}] - {}", mv.version, mv.status, mv.changes);
        }
    }
    println!();
    registry
}

pub fn check_compatibility_matrix(registry: &mut VersionRegistry, ctx: &mut RecipeContext) {
    println!("--- Section 2: Compatibility Matrix ---");
    let v = SemanticVersion::new;
    registry.add_compatibility_rule(CompatibilityRule {
        source_model: "fraud-detector".into(),
        source_range: VersionRange::new(v(1, 0, 0), v(1, 2, 0)),
        target_model: "feature-encoder".into(),
        target_range: VersionRange::new(v(1, 0, 0), v(1, 0, 0)),
        compatible: true,
    });
    registry.add_compatibility_rule(CompatibilityRule {
        source_model: "fraud-detector".into(),
        source_range: VersionRange::new(v(2, 0, 0), v(3, 1, 0)),
        target_model: "feature-encoder".into(),
        target_range: VersionRange::new(v(2, 0, 0), v(2, 1, 0)),
        compatible: true,
    });
    let checks = [
        (v(1, 1, 0), v(1, 0, 0)),
        (v(2, 0, 0), v(1, 0, 0)),
        (v(2, 0, 0), v(2, 0, 0)),
        (v(3, 0, 0), v(2, 1, 0)),
    ];
    let mut cc = 0i64;
    for (sv, tv) in &checks {
        let ok = registry.check_compatibility("fraud-detector", sv, "feature-encoder", tv);
        if ok {
            cc += 1;
        }
        println!(
            "  fd v{sv} + fe v{tv} => {}",
            if ok { "COMPATIBLE" } else { "INCOMPATIBLE" }
        );
    }
    ctx.record_metric("compatible_pairs", cc);
    println!();
}

pub fn resolve_dependencies(registry: &mut VersionRegistry, ctx: &mut RecipeContext) {
    println!("--- Section 3: Dependency Resolution ---");
    let v = SemanticVersion::new;
    registry.add_dependency(
        "fraud-detector",
        &v(2, 0, 0),
        "feature-encoder",
        VersionRange::new(v(2, 0, 0), v(2, 1, 0)),
    );
    registry.add_dependency(
        "fraud-detector",
        &v(3, 0, 0),
        "feature-encoder",
        VersionRange::new(v(2, 1, 0), v(2, 1, 0)),
    );
    for ver in [v(1, 0, 0), v(2, 0, 0), v(3, 0, 0)] {
        let resolved = registry.resolve_dependencies("fraud-detector", &ver);
        if resolved.is_empty() {
            println!("  fd v{ver} => no deps");
        } else {
            for (dm, dv) in &resolved {
                println!("  fd v{ver} requires {dm} v{dv}");
            }
        }
    }
    ctx.record_metric("dependency_rules", registry.dependencies.len() as i64);
    println!();
}

pub fn compute_upgrade_paths(registry: &VersionRegistry) {
    println!("--- Section 4: Upgrade Paths ---");
    let v = SemanticVersion::new;
    for (model, from, to) in [
        ("fraud-detector", v(1, 0, 0), v(2, 1, 0)),
        ("fraud-detector", v(1, 2, 0), v(3, 0, 0)),
        ("feature-encoder", v(1, 0, 0), v(2, 1, 0)),
    ] {
        match registry.find_upgrade_path(model, &from, &to) {
            Some(path) => {
                println!(
                    "  {model} v{from} -> v{to}: {} steps, {} breaking",
                    path.steps.len(),
                    path.breaking_changes
                );
                for note in &path.migration_notes {
                    println!("    - {note}");
                }
            }
            None => println!("  {model} v{from} -> v{to}: no path"),
        }
    }
    println!();
}

pub fn apply_lifecycle_transitions(registry: &mut VersionRegistry) {
    println!("--- Section 5: Lifecycle Transitions ---");
    let v = SemanticVersion::new;
    for (model, ver, from, to) in [
        ("fraud-detector", v(3, 1, 0), "alpha", VersionStatus::Beta),
        ("fraud-detector", v(3, 0, 0), "beta", VersionStatus::Stable),
        (
            "fraud-detector",
            v(1, 2, 0),
            "deprecated",
            VersionStatus::Archived,
        ),
    ] {
        match registry.transition_status(model, &ver, to.clone()) {
            Ok(()) => println!("  {model} v{ver}: {from} -> {to} [OK]"),
            Err(msg) => println!("  {model} v{ver}: {from} -> {to} [FAILED: {msg}]"),
        }
    }
    let inv = registry.transition_status("fraud-detector", &v(1, 0, 0), VersionStatus::Beta);
    println!(
        "  fd v1.0.0: archived -> beta [{}]",
        if inv.is_err() { "BLOCKED" } else { "OK" }
    );
    println!();
}

pub fn print_version_history(registry: &VersionRegistry, ctx: &mut RecipeContext) {
    println!("--- Section 6: Version History ---");
    for mn in ["fraud-detector", "feature-encoder"] {
        if let Some(mvs) = registry.versions.get(mn) {
            println!("  {mn}:");
            for mv in mvs {
                let brk = if mv.version.major > 1
                    && mv
                        .parent_version
                        .as_ref()
                        .is_some_and(|p| p.major < mv.version.major)
                {
                    " [BREAKING]"
                } else {
                    ""
                };
                println!("    v{} ({}) - {}{brk}", mv.version, mv.status, mv.changes);
            }
        }
    }
    let diff = registry.version_diff(
        "fraud-detector",
        &SemanticVersion::new(1, 0, 0),
        &SemanticVersion::new(3, 0, 0),
    );
    println!(
        "  Diff fd v1.0.0->v3.0.0: {} intermediate, {} breaking, {} -> {}",
        diff.intermediate_count, diff.breaking_count, diff.from_status, diff.to_status
    );
    ctx.record_metric("models_tracked", registry.versions.len() as i64);
    ctx.record_metric(
        "compatibility_rules",
        registry.compatibility_rules.len() as i64,
    );
    println!();
}
