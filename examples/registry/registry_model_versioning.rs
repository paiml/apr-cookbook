//! # Recipe: Model Version Management
//!
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

fn register_model_versions(ctx: &mut RecipeContext) -> VersionRegistry {
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

fn check_compatibility_matrix(registry: &mut VersionRegistry, ctx: &mut RecipeContext) {
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

fn resolve_dependencies(registry: &mut VersionRegistry, ctx: &mut RecipeContext) {
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

fn compute_upgrade_paths(registry: &VersionRegistry) {
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

fn apply_lifecycle_transitions(registry: &mut VersionRegistry) {
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

fn print_version_history(registry: &VersionRegistry, ctx: &mut RecipeContext) {
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

// --- Core Types ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SemanticVersion {
    major: u32,
    minor: u32,
    patch: u32,
}
impl SemanticVersion {
    const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    fn parse(s: &str) -> Option<Self> {
        let p: Vec<&str> = s.split('.').collect();
        if p.len() != 3 {
            return None;
        }
        Some(Self {
            major: p[0].parse().ok()?,
            minor: p[1].parse().ok()?,
            patch: p[2].parse().ok()?,
        })
    }
    fn is_breaking_from(&self, other: &Self) -> bool {
        self.major > other.major
    }
}
impl Ord for SemanticVersion {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&o.major)
            .then(self.minor.cmp(&o.minor))
            .then(self.patch.cmp(&o.patch))
    }
}
impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VersionStatus {
    Alpha,
    Beta,
    Stable,
    Deprecated,
    Archived,
}
impl VersionStatus {
    fn can_transition_to(&self, target: &Self) -> bool {
        matches!(
            (self, target),
            (Self::Alpha, Self::Beta | Self::Deprecated)
                | (Self::Beta, Self::Stable | Self::Deprecated)
                | (Self::Stable, Self::Deprecated)
                | (Self::Deprecated, Self::Archived)
        )
    }
}
impl fmt::Display for VersionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Alpha => "alpha",
                Self::Beta => "beta",
                Self::Stable => "stable",
                Self::Deprecated => "deprecated",
                Self::Archived => "archived",
            }
        )
    }
}

#[derive(Debug, Clone)]
struct ModelVersion {
    name: String,
    version: SemanticVersion,
    status: VersionStatus,
    parent_version: Option<SemanticVersion>,
    changes: String,
}

#[derive(Debug, Clone)]
struct VersionRange {
    min: SemanticVersion,
    max: SemanticVersion,
}
impl VersionRange {
    const fn new(min: SemanticVersion, max: SemanticVersion) -> Self {
        Self { min, max }
    }
    fn contains(&self, v: &SemanticVersion) -> bool {
        v >= &self.min && v <= &self.max
    }
}

#[derive(Debug, Clone)]
struct CompatibilityRule {
    source_model: String,
    source_range: VersionRange,
    target_model: String,
    target_range: VersionRange,
    compatible: bool,
}

#[derive(Debug, Clone)]
struct UpgradePath {
    steps: Vec<SemanticVersion>,
    breaking_changes: usize,
    migration_notes: Vec<String>,
}

#[derive(Debug)]
struct VersionDiff {
    intermediate_count: usize,
    breaking_count: usize,
    from_status: VersionStatus,
    to_status: VersionStatus,
}

#[derive(Debug, Clone)]
struct DependencyRule {
    model: String,
    version: SemanticVersion,
    depends_on_model: String,
    depends_on_range: VersionRange,
}

#[derive(Debug)]
struct VersionRegistry {
    versions: HashMap<String, Vec<ModelVersion>>,
    compatibility_rules: Vec<CompatibilityRule>,
    dependencies: Vec<DependencyRule>,
}
impl VersionRegistry {
    fn new() -> Self {
        Self {
            versions: HashMap::new(),
            compatibility_rules: Vec::new(),
            dependencies: Vec::new(),
        }
    }
    fn register(&mut self, mv: ModelVersion) {
        let e = self.versions.entry(mv.name.clone()).or_default();
        e.push(mv);
        e.sort_by(|a, b| a.version.cmp(&b.version));
    }
    fn add_compatibility_rule(&mut self, rule: CompatibilityRule) {
        self.compatibility_rules.push(rule);
    }
    fn add_dependency(
        &mut self,
        model: &str,
        ver: &SemanticVersion,
        dep_model: &str,
        dep_range: VersionRange,
    ) {
        self.dependencies.push(DependencyRule {
            model: model.into(),
            version: ver.clone(),
            depends_on_model: dep_model.into(),
            depends_on_range: dep_range,
        });
    }
    fn check_compatibility(
        &self,
        sm: &str,
        sv: &SemanticVersion,
        tm: &str,
        tv: &SemanticVersion,
    ) -> bool {
        self.compatibility_rules.iter().any(|r| {
            r.compatible
                && r.source_model == sm
                && r.target_model == tm
                && r.source_range.contains(sv)
                && r.target_range.contains(tv)
        })
    }
    fn resolve_dependencies(
        &self,
        model: &str,
        ver: &SemanticVersion,
    ) -> Vec<(String, SemanticVersion)> {
        let mut res = Vec::new();
        for dep in &self.dependencies {
            if dep.model == model && dep.version == *ver {
                if let Some(dvs) = self.versions.get(&dep.depends_on_model) {
                    if let Some(best) = dvs
                        .iter()
                        .filter(|mv| dep.depends_on_range.contains(&mv.version))
                        .max_by(|a, b| a.version.cmp(&b.version))
                    {
                        res.push((dep.depends_on_model.clone(), best.version.clone()));
                    }
                }
            }
        }
        res
    }
    fn find_upgrade_path(
        &self,
        model: &str,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Option<UpgradePath> {
        let mvs = self.versions.get(model)?;
        let steps: Vec<SemanticVersion> = mvs
            .iter()
            .filter(|mv| mv.version >= *from && mv.version <= *to)
            .map(|mv| mv.version.clone())
            .collect();
        if steps.len() < 2 {
            return None;
        }
        let mut bc = 0;
        let mut notes = Vec::new();
        for w in steps.windows(2) {
            if w[1].is_breaking_from(&w[0]) {
                bc += 1;
                if let Some(mv) = mvs.iter().find(|mv| mv.version == w[1]) {
                    notes.push(format!("v{}: {}", mv.version, mv.changes));
                }
            }
        }
        Some(UpgradePath {
            steps,
            breaking_changes: bc,
            migration_notes: notes,
        })
    }
    fn transition_status(
        &mut self,
        model: &str,
        ver: &SemanticVersion,
        new_status: VersionStatus,
    ) -> std::result::Result<(), String> {
        let mvs = self
            .versions
            .get_mut(model)
            .ok_or_else(|| format!("model '{model}' not found"))?;
        let mv = mvs
            .iter_mut()
            .find(|mv| mv.version == *ver)
            .ok_or_else(|| format!("version {ver} not found for '{model}'"))?;
        if !mv.status.can_transition_to(&new_status) {
            return Err(format!(
                "cannot transition from {} to {}",
                mv.status, new_status
            ));
        }
        mv.status = new_status;
        Ok(())
    }
    fn version_diff(
        &self,
        model: &str,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> VersionDiff {
        let empty = Vec::new();
        let mvs = self.versions.get(model).unwrap_or(&empty);
        VersionDiff {
            intermediate_count: mvs
                .iter()
                .filter(|mv| mv.version > *from && mv.version < *to)
                .count(),
            breaking_count: mvs
                .iter()
                .filter(|mv| mv.version > *from && mv.version <= *to)
                .filter(|mv| {
                    mv.parent_version
                        .as_ref()
                        .is_some_and(|p| mv.version.is_breaking_from(p))
                })
                .count(),
            from_status: mvs
                .iter()
                .find(|mv| mv.version == *from)
                .map_or(VersionStatus::Alpha, |mv| mv.status.clone()),
            to_status: mvs
                .iter()
                .find(|mv| mv.version == *to)
                .map_or(VersionStatus::Alpha, |mv| mv.status.clone()),
        }
    }
}

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
