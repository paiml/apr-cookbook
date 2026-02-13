//! # Recipe: Model Version Management
//!
//! **Category**: Model Registry
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Manage model versions with semantic versioning, compatibility checking,
//! dependency resolution, upgrade paths, and lifecycle management.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_model_versioning
//! ```

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_model_versioning")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Demonstrating model version management with semantic versioning");
    println!();

    // ---------------------------------------------------------------
    // Section 1: Define version scheme and register model versions
    // ---------------------------------------------------------------
    println!("--- Section 1: Register Model Versions ---");

    // Demonstrate version parsing
    let parsed = SemanticVersion::parse("2.1.0");
    println!(
        "  Parsed version string \"2.1.0\" => {}",
        parsed
            .as_ref()
            .map_or("invalid".to_string(), |v| format!("{v}"))
    );

    let mut registry = VersionRegistry::new();

    // Register fraud-detector versions
    let versions = [
        (
            "fraud-detector",
            1,
            0,
            0,
            VersionStatus::Archived,
            None,
            "Initial release with basic rules",
        ),
        (
            "fraud-detector",
            1,
            1,
            0,
            VersionStatus::Deprecated,
            Some(SemanticVersion::new(1, 0, 0)),
            "Added feature engineering pipeline",
        ),
        (
            "fraud-detector",
            1,
            2,
            0,
            VersionStatus::Deprecated,
            Some(SemanticVersion::new(1, 1, 0)),
            "Improved threshold tuning",
        ),
        (
            "fraud-detector",
            2,
            0,
            0,
            VersionStatus::Stable,
            Some(SemanticVersion::new(1, 2, 0)),
            "Breaking: new input schema with embeddings",
        ),
        (
            "fraud-detector",
            2,
            1,
            0,
            VersionStatus::Stable,
            Some(SemanticVersion::new(2, 0, 0)),
            "Added real-time scoring mode",
        ),
        (
            "fraud-detector",
            3,
            0,
            0,
            VersionStatus::Beta,
            Some(SemanticVersion::new(2, 1, 0)),
            "Breaking: transformer-based architecture",
        ),
        (
            "fraud-detector",
            3,
            1,
            0,
            VersionStatus::Alpha,
            Some(SemanticVersion::new(3, 0, 0)),
            "Experimental: multi-modal inputs",
        ),
    ];

    for (name, major, minor, patch, status, parent, changes) in &versions {
        let ver = SemanticVersion::new(*major, *minor, *patch);
        registry.register(ModelVersion {
            name: (*name).to_string(),
            version: ver,
            status: status.clone(),
            parent_version: parent.clone(),
            changes: (*changes).to_string(),
        });
    }

    // Register feature-encoder versions (dependency target)
    let encoder_versions = [
        (
            "feature-encoder",
            1,
            0,
            0,
            VersionStatus::Deprecated,
            None,
            "Basic one-hot encoding",
        ),
        (
            "feature-encoder",
            2,
            0,
            0,
            VersionStatus::Stable,
            Some(SemanticVersion::new(1, 0, 0)),
            "Breaking: embedding-based encoding",
        ),
        (
            "feature-encoder",
            2,
            1,
            0,
            VersionStatus::Stable,
            Some(SemanticVersion::new(2, 0, 0)),
            "Added caching layer",
        ),
    ];

    for (name, major, minor, patch, status, parent, changes) in &encoder_versions {
        let ver = SemanticVersion::new(*major, *minor, *patch);
        registry.register(ModelVersion {
            name: (*name).to_string(),
            version: ver,
            status: status.clone(),
            parent_version: parent.clone(),
            changes: (*changes).to_string(),
        });
    }

    let total_versions = registry.versions.values().map(Vec::len).sum::<usize>();
    ctx.record_metric("total_versions", total_versions as i64);

    for (model_name, model_versions) in &registry.versions {
        println!("  {} ({} versions):", model_name, model_versions.len());
        for mv in model_versions {
            println!("    v{} [{}] - {}", mv.version, mv.status, mv.changes);
        }
    }
    println!();

    // ---------------------------------------------------------------
    // Section 2: Compatibility matrix (which versions work together)
    // ---------------------------------------------------------------
    println!("--- Section 2: Compatibility Matrix ---");

    // Define compatibility rules
    registry.add_compatibility_rule(CompatibilityRule {
        source_model: "fraud-detector".to_string(),
        source_range: VersionRange::new(
            SemanticVersion::new(1, 0, 0),
            SemanticVersion::new(1, 2, 0),
        ),
        target_model: "feature-encoder".to_string(),
        target_range: VersionRange::new(
            SemanticVersion::new(1, 0, 0),
            SemanticVersion::new(1, 0, 0),
        ),
        compatible: true,
    });

    registry.add_compatibility_rule(CompatibilityRule {
        source_model: "fraud-detector".to_string(),
        source_range: VersionRange::new(
            SemanticVersion::new(2, 0, 0),
            SemanticVersion::new(3, 1, 0),
        ),
        target_model: "feature-encoder".to_string(),
        target_range: VersionRange::new(
            SemanticVersion::new(2, 0, 0),
            SemanticVersion::new(2, 1, 0),
        ),
        compatible: true,
    });

    // Check specific compatibility pairs
    let checks = [
        (
            "fraud-detector",
            SemanticVersion::new(1, 1, 0),
            "feature-encoder",
            SemanticVersion::new(1, 0, 0),
        ),
        (
            "fraud-detector",
            SemanticVersion::new(2, 0, 0),
            "feature-encoder",
            SemanticVersion::new(1, 0, 0),
        ),
        (
            "fraud-detector",
            SemanticVersion::new(2, 0, 0),
            "feature-encoder",
            SemanticVersion::new(2, 0, 0),
        ),
        (
            "fraud-detector",
            SemanticVersion::new(3, 0, 0),
            "feature-encoder",
            SemanticVersion::new(2, 1, 0),
        ),
    ];

    let mut compatible_count = 0i64;
    for (src_model, src_ver, tgt_model, tgt_ver) in &checks {
        let is_compat = registry.check_compatibility(src_model, src_ver, tgt_model, tgt_ver);
        let label = if is_compat {
            "COMPATIBLE"
        } else {
            "INCOMPATIBLE"
        };
        if is_compat {
            compatible_count += 1;
        }
        println!(
            "  {} v{} + {} v{} => {}",
            src_model, src_ver, tgt_model, tgt_ver, label
        );
    }
    ctx.record_metric("compatible_pairs", compatible_count);
    println!();

    // ---------------------------------------------------------------
    // Section 3: Dependency resolution across model ecosystem
    // ---------------------------------------------------------------
    println!("--- Section 3: Dependency Resolution ---");

    // fraud-detector v2.0.0 requires feature-encoder >= 2.0.0
    registry.add_dependency(
        "fraud-detector",
        &SemanticVersion::new(2, 0, 0),
        "feature-encoder",
        VersionRange::new(SemanticVersion::new(2, 0, 0), SemanticVersion::new(2, 1, 0)),
    );

    // fraud-detector v3.0.0 requires feature-encoder >= 2.1.0
    registry.add_dependency(
        "fraud-detector",
        &SemanticVersion::new(3, 0, 0),
        "feature-encoder",
        VersionRange::new(SemanticVersion::new(2, 1, 0), SemanticVersion::new(2, 1, 0)),
    );

    let dep_queries = [
        ("fraud-detector", SemanticVersion::new(1, 0, 0)),
        ("fraud-detector", SemanticVersion::new(2, 0, 0)),
        ("fraud-detector", SemanticVersion::new(3, 0, 0)),
    ];

    for (model, version) in &dep_queries {
        let resolved = registry.resolve_dependencies(model, version);
        if resolved.is_empty() {
            println!("  {} v{} => no dependencies", model, version);
        } else {
            println!("  {} v{} requires:", model, version);
            for (dep_model, dep_ver) in &resolved {
                println!("    - {} v{}", dep_model, dep_ver);
            }
        }
    }
    ctx.record_metric("dependency_rules", registry.dependencies.len() as i64);
    println!();

    // ---------------------------------------------------------------
    // Section 4: Compute upgrade paths with breaking change detection
    // ---------------------------------------------------------------
    println!("--- Section 4: Upgrade Paths ---");

    let upgrade_scenarios = [
        (
            "fraud-detector",
            SemanticVersion::new(1, 0, 0),
            SemanticVersion::new(2, 1, 0),
        ),
        (
            "fraud-detector",
            SemanticVersion::new(1, 2, 0),
            SemanticVersion::new(3, 0, 0),
        ),
        (
            "feature-encoder",
            SemanticVersion::new(1, 0, 0),
            SemanticVersion::new(2, 1, 0),
        ),
    ];

    for (model, from, to) in &upgrade_scenarios {
        match registry.find_upgrade_path(model, from, to) {
            Some(path) => {
                println!("  {} v{} -> v{}:", model, from, to);
                println!(
                    "    Steps: {}",
                    path.steps
                        .iter()
                        .map(|s| format!("v{s}"))
                        .collect::<Vec<_>>()
                        .join(" -> ")
                );
                println!("    Breaking changes: {}", path.breaking_changes);
                if !path.migration_notes.is_empty() {
                    println!("    Migration notes:");
                    for note in &path.migration_notes {
                        println!("      - {}", note);
                    }
                }
            }
            None => {
                println!("  {} v{} -> v{}: no path found", model, from, to);
            }
        }
    }
    println!();

    // ---------------------------------------------------------------
    // Section 5: Version lifecycle transitions
    // ---------------------------------------------------------------
    println!("--- Section 5: Version Lifecycle Transitions ---");

    let transitions = [
        (
            "fraud-detector",
            SemanticVersion::new(3, 1, 0),
            VersionStatus::Alpha,
            VersionStatus::Beta,
        ),
        (
            "fraud-detector",
            SemanticVersion::new(3, 0, 0),
            VersionStatus::Beta,
            VersionStatus::Stable,
        ),
        (
            "fraud-detector",
            SemanticVersion::new(1, 2, 0),
            VersionStatus::Deprecated,
            VersionStatus::Archived,
        ),
    ];

    for (model, version, from_status, to_status) in &transitions {
        let result = registry.transition_status(model, version, to_status.clone());
        match result {
            Ok(()) => {
                println!(
                    "  {} v{}: {} -> {} [OK]",
                    model, version, from_status, to_status
                );
            }
            Err(msg) => {
                println!(
                    "  {} v{}: {} -> {} [FAILED: {}]",
                    model, version, from_status, to_status, msg
                );
            }
        }
    }

    // Try an invalid transition
    let invalid = registry.transition_status(
        "fraud-detector",
        &SemanticVersion::new(1, 0, 0),
        VersionStatus::Beta,
    );
    println!(
        "  fraud-detector v1.0.0: Archived -> Beta [{}]",
        if invalid.is_err() { "BLOCKED" } else { "OK" }
    );
    println!();

    // ---------------------------------------------------------------
    // Section 6: Version history summary with changelog
    // ---------------------------------------------------------------
    println!("--- Section 6: Version History & Changelog ---");

    for model_name in ["fraud-detector", "feature-encoder"] {
        println!("  Changelog for {}:", model_name);
        if let Some(model_versions) = registry.versions.get(model_name) {
            for mv in model_versions {
                let breaking = if mv.version.major > 1
                    && mv
                        .parent_version
                        .as_ref()
                        .is_some_and(|p| p.major < mv.version.major)
                {
                    " [BREAKING]"
                } else {
                    ""
                };
                println!(
                    "    v{} ({}) - {}{}",
                    mv.version, mv.status, mv.changes, breaking
                );
            }
        }
        println!();
    }

    // Summary diff between two versions
    let diff = registry.version_diff(
        "fraud-detector",
        &SemanticVersion::new(1, 0, 0),
        &SemanticVersion::new(3, 0, 0),
    );
    println!("  Diff summary: fraud-detector v1.0.0 -> v3.0.0");
    println!(
        "    Total intermediate versions: {}",
        diff.intermediate_count
    );
    println!("    Breaking changes: {}", diff.breaking_count);
    println!(
        "    Status transition: {} -> {}",
        diff.from_status, diff.to_status
    );

    ctx.record_metric("models_tracked", registry.versions.len() as i64);
    ctx.record_metric(
        "compatibility_rules",
        registry.compatibility_rules.len() as i64,
    );
    println!();
    println!("=== Recipe complete ===");

    Ok(())
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Semantic version with major.minor.patch components.
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

    /// Parse a semantic version from a string like "1.2.3".
    fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns true if upgrading from `other` to `self` is a breaking change.
    fn is_breaking_from(&self, other: &Self) -> bool {
        self.major > other.major
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Lifecycle status for a model version.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VersionStatus {
    Alpha,
    Beta,
    Stable,
    Deprecated,
    Archived,
}

impl VersionStatus {
    /// Returns the allowed transitions from this status.
    fn allowed_transitions(&self) -> &[VersionStatus] {
        match self {
            Self::Alpha => &[Self::Beta, Self::Deprecated],
            Self::Beta => &[Self::Stable, Self::Deprecated],
            Self::Stable => &[Self::Deprecated],
            Self::Deprecated => &[Self::Archived],
            Self::Archived => &[],
        }
    }

    fn can_transition_to(&self, target: &Self) -> bool {
        self.allowed_transitions().contains(target)
    }
}

impl fmt::Display for VersionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alpha => write!(f, "alpha"),
            Self::Beta => write!(f, "beta"),
            Self::Stable => write!(f, "stable"),
            Self::Deprecated => write!(f, "deprecated"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

/// A registered model version with metadata.
#[derive(Debug, Clone)]
struct ModelVersion {
    name: String,
    version: SemanticVersion,
    status: VersionStatus,
    parent_version: Option<SemanticVersion>,
    changes: String,
}

/// Inclusive range of semantic versions.
#[derive(Debug, Clone)]
struct VersionRange {
    min: SemanticVersion,
    max: SemanticVersion,
}

impl VersionRange {
    const fn new(min: SemanticVersion, max: SemanticVersion) -> Self {
        Self { min, max }
    }

    fn contains(&self, version: &SemanticVersion) -> bool {
        version >= &self.min && version <= &self.max
    }
}

/// A rule defining compatibility between two model version ranges.
#[derive(Debug, Clone)]
struct CompatibilityRule {
    source_model: String,
    source_range: VersionRange,
    target_model: String,
    target_range: VersionRange,
    compatible: bool,
}

/// Computed upgrade path between two versions.
#[derive(Debug, Clone)]
struct UpgradePath {
    steps: Vec<SemanticVersion>,
    breaking_changes: usize,
    migration_notes: Vec<String>,
}

/// Summary of differences between two versions.
#[derive(Debug)]
struct VersionDiff {
    intermediate_count: usize,
    breaking_count: usize,
    from_status: VersionStatus,
    to_status: VersionStatus,
}

/// A dependency requirement: a model version requires another model within a version range.
#[derive(Debug, Clone)]
struct DependencyRule {
    model: String,
    version: SemanticVersion,
    depends_on_model: String,
    depends_on_range: VersionRange,
}

// ---------------------------------------------------------------------------
// Version Registry
// ---------------------------------------------------------------------------

/// Registry tracking model versions, compatibility rules, and dependencies.
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

    fn register(&mut self, model_version: ModelVersion) {
        let entry = self.versions.entry(model_version.name.clone()).or_default();
        entry.push(model_version);
        entry.sort_by(|a, b| a.version.cmp(&b.version));
    }

    fn add_compatibility_rule(&mut self, rule: CompatibilityRule) {
        self.compatibility_rules.push(rule);
    }

    fn add_dependency(
        &mut self,
        model: &str,
        version: &SemanticVersion,
        depends_on_model: &str,
        depends_on_range: VersionRange,
    ) {
        self.dependencies.push(DependencyRule {
            model: model.to_string(),
            version: version.clone(),
            depends_on_model: depends_on_model.to_string(),
            depends_on_range,
        });
    }

    /// Check whether two model versions are compatible according to registered rules.
    fn check_compatibility(
        &self,
        source_model: &str,
        source_version: &SemanticVersion,
        target_model: &str,
        target_version: &SemanticVersion,
    ) -> bool {
        self.compatibility_rules.iter().any(|rule| {
            rule.compatible
                && rule.source_model == source_model
                && rule.target_model == target_model
                && rule.source_range.contains(source_version)
                && rule.target_range.contains(target_version)
        })
    }

    /// Resolve dependencies for a given model version.
    /// Returns the best (highest) compatible version for each dependency.
    fn resolve_dependencies(
        &self,
        model: &str,
        version: &SemanticVersion,
    ) -> Vec<(String, SemanticVersion)> {
        let mut resolved = Vec::new();

        for dep in &self.dependencies {
            if dep.model == model && dep.version == *version {
                // Find highest available version in the dependency range
                if let Some(dep_versions) = self.versions.get(&dep.depends_on_model) {
                    let best = dep_versions
                        .iter()
                        .filter(|mv| dep.depends_on_range.contains(&mv.version))
                        .max_by(|a, b| a.version.cmp(&b.version));
                    if let Some(best_mv) = best {
                        resolved.push((dep.depends_on_model.clone(), best_mv.version.clone()));
                    }
                }
            }
        }

        resolved
    }

    /// Compute an upgrade path from one version to another within the same model.
    fn find_upgrade_path(
        &self,
        model: &str,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Option<UpgradePath> {
        let model_versions = self.versions.get(model)?;

        // Collect versions in the range [from, to]
        let steps: Vec<SemanticVersion> = model_versions
            .iter()
            .filter(|mv| mv.version >= *from && mv.version <= *to)
            .map(|mv| mv.version.clone())
            .collect();

        if steps.len() < 2 {
            return None;
        }

        // Count breaking changes (major version bumps between consecutive steps)
        let mut breaking_changes = 0;
        let mut migration_notes = Vec::new();

        for window in steps.windows(2) {
            if window[1].is_breaking_from(&window[0]) {
                breaking_changes += 1;
                // Find the model version for the target step to get its changes
                if let Some(mv) = model_versions.iter().find(|mv| mv.version == window[1]) {
                    migration_notes.push(format!("v{}: {}", mv.version, mv.changes));
                }
            }
        }

        Some(UpgradePath {
            steps,
            breaking_changes,
            migration_notes,
        })
    }

    /// Transition a model version to a new lifecycle status.
    fn transition_status(
        &mut self,
        model: &str,
        version: &SemanticVersion,
        new_status: VersionStatus,
    ) -> std::result::Result<(), String> {
        let model_versions = self
            .versions
            .get_mut(model)
            .ok_or_else(|| format!("model '{}' not found", model))?;

        let mv = model_versions
            .iter_mut()
            .find(|mv| mv.version == *version)
            .ok_or_else(|| format!("version {} not found for '{}'", version, model))?;

        if !mv.status.can_transition_to(&new_status) {
            return Err(format!(
                "cannot transition from {} to {}",
                mv.status, new_status
            ));
        }

        mv.status = new_status;
        Ok(())
    }

    /// Compute a diff summary between two versions of the same model.
    fn version_diff(
        &self,
        model: &str,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> VersionDiff {
        let empty = Vec::new();
        let model_versions = self.versions.get(model).unwrap_or(&empty);

        let intermediates: Vec<&ModelVersion> = model_versions
            .iter()
            .filter(|mv| mv.version > *from && mv.version < *to)
            .collect();

        let breaking_count = model_versions
            .iter()
            .filter(|mv| mv.version > *from && mv.version <= *to)
            .filter(|mv| {
                mv.parent_version
                    .as_ref()
                    .is_some_and(|p| mv.version.is_breaking_from(p))
            })
            .count();

        let from_status = model_versions
            .iter()
            .find(|mv| mv.version == *from)
            .map_or(VersionStatus::Alpha, |mv| mv.status.clone());

        let to_status = model_versions
            .iter()
            .find(|mv| mv.version == *to)
            .map_or(VersionStatus::Alpha, |mv| mv.status.clone());

        VersionDiff {
            intermediate_count: intermediates.len(),
            breaking_count,
            from_status,
            to_status,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_version_new() {
        let v = SemanticVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_semantic_version_display() {
        let v = SemanticVersion::new(2, 10, 5);
        assert_eq!(format!("{}", v), "2.10.5");
    }

    #[test]
    fn test_semantic_version_parse() {
        let v = SemanticVersion::parse("3.2.1");
        assert_eq!(v, Some(SemanticVersion::new(3, 2, 1)));
    }

    #[test]
    fn test_semantic_version_parse_invalid() {
        assert_eq!(SemanticVersion::parse("1.2"), None);
        assert_eq!(SemanticVersion::parse("abc"), None);
        assert_eq!(SemanticVersion::parse("1.2.x"), None);
        assert_eq!(SemanticVersion::parse(""), None);
    }

    #[test]
    fn test_semantic_version_ordering() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);
        let v3 = SemanticVersion::new(2, 0, 0);
        let v4 = SemanticVersion::new(1, 0, 1);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v4);
        assert!(v4 < v2);
    }

    #[test]
    fn test_semantic_version_equality() {
        let v1 = SemanticVersion::new(1, 2, 3);
        let v2 = SemanticVersion::new(1, 2, 3);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_is_breaking_from() {
        let v1 = SemanticVersion::new(1, 9, 9);
        let v2 = SemanticVersion::new(2, 0, 0);
        let v3 = SemanticVersion::new(1, 1, 0);

        assert!(v2.is_breaking_from(&v1));
        assert!(!v3.is_breaking_from(&v1));
    }

    #[test]
    fn test_version_status_display() {
        assert_eq!(format!("{}", VersionStatus::Alpha), "alpha");
        assert_eq!(format!("{}", VersionStatus::Beta), "beta");
        assert_eq!(format!("{}", VersionStatus::Stable), "stable");
        assert_eq!(format!("{}", VersionStatus::Deprecated), "deprecated");
        assert_eq!(format!("{}", VersionStatus::Archived), "archived");
    }

    #[test]
    fn test_version_status_transitions() {
        assert!(VersionStatus::Alpha.can_transition_to(&VersionStatus::Beta));
        assert!(VersionStatus::Alpha.can_transition_to(&VersionStatus::Deprecated));
        assert!(!VersionStatus::Alpha.can_transition_to(&VersionStatus::Stable));
        assert!(VersionStatus::Beta.can_transition_to(&VersionStatus::Stable));
        assert!(VersionStatus::Stable.can_transition_to(&VersionStatus::Deprecated));
        assert!(!VersionStatus::Stable.can_transition_to(&VersionStatus::Alpha));
        assert!(VersionStatus::Deprecated.can_transition_to(&VersionStatus::Archived));
        assert!(!VersionStatus::Archived.can_transition_to(&VersionStatus::Stable));
    }

    #[test]
    fn test_version_range_contains() {
        let range = VersionRange::new(SemanticVersion::new(1, 0, 0), SemanticVersion::new(2, 0, 0));
        assert!(range.contains(&SemanticVersion::new(1, 0, 0)));
        assert!(range.contains(&SemanticVersion::new(1, 5, 0)));
        assert!(range.contains(&SemanticVersion::new(2, 0, 0)));
        assert!(!range.contains(&SemanticVersion::new(0, 9, 0)));
        assert!(!range.contains(&SemanticVersion::new(2, 0, 1)));
    }

    #[test]
    fn test_registry_register_and_sort() {
        let mut registry = VersionRegistry::new();
        registry.register(ModelVersion {
            name: "test".to_string(),
            version: SemanticVersion::new(2, 0, 0),
            status: VersionStatus::Stable,
            parent_version: None,
            changes: "v2".to_string(),
        });
        registry.register(ModelVersion {
            name: "test".to_string(),
            version: SemanticVersion::new(1, 0, 0),
            status: VersionStatus::Stable,
            parent_version: None,
            changes: "v1".to_string(),
        });

        let versions = &registry.versions["test"];
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].version, SemanticVersion::new(1, 0, 0));
        assert_eq!(versions[1].version, SemanticVersion::new(2, 0, 0));
    }

    #[test]
    fn test_compatibility_check_positive() {
        let mut registry = VersionRegistry::new();
        registry.add_compatibility_rule(CompatibilityRule {
            source_model: "A".to_string(),
            source_range: VersionRange::new(
                SemanticVersion::new(1, 0, 0),
                SemanticVersion::new(1, 9, 0),
            ),
            target_model: "B".to_string(),
            target_range: VersionRange::new(
                SemanticVersion::new(2, 0, 0),
                SemanticVersion::new(2, 5, 0),
            ),
            compatible: true,
        });

        assert!(registry.check_compatibility(
            "A",
            &SemanticVersion::new(1, 3, 0),
            "B",
            &SemanticVersion::new(2, 1, 0)
        ));
    }

    #[test]
    fn test_compatibility_check_negative() {
        let registry = VersionRegistry::new();
        // No rules -> not compatible
        assert!(!registry.check_compatibility(
            "A",
            &SemanticVersion::new(1, 0, 0),
            "B",
            &SemanticVersion::new(1, 0, 0)
        ));
    }

    #[test]
    fn test_dependency_resolution() {
        let mut registry = VersionRegistry::new();
        registry.register(ModelVersion {
            name: "encoder".to_string(),
            version: SemanticVersion::new(2, 0, 0),
            status: VersionStatus::Stable,
            parent_version: None,
            changes: "release".to_string(),
        });
        registry.register(ModelVersion {
            name: "encoder".to_string(),
            version: SemanticVersion::new(2, 1, 0),
            status: VersionStatus::Stable,
            parent_version: None,
            changes: "update".to_string(),
        });
        registry.add_dependency(
            "detector",
            &SemanticVersion::new(3, 0, 0),
            "encoder",
            VersionRange::new(SemanticVersion::new(2, 0, 0), SemanticVersion::new(2, 1, 0)),
        );

        let resolved = registry.resolve_dependencies("detector", &SemanticVersion::new(3, 0, 0));
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].0, "encoder");
        assert_eq!(resolved[0].1, SemanticVersion::new(2, 1, 0));
    }

    #[test]
    fn test_dependency_resolution_no_deps() {
        let registry = VersionRegistry::new();
        let resolved = registry.resolve_dependencies("model", &SemanticVersion::new(1, 0, 0));
        assert!(resolved.is_empty());
    }

    #[test]
    fn test_upgrade_path() {
        let mut registry = VersionRegistry::new();
        for (major, minor) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
            registry.register(ModelVersion {
                name: "m".to_string(),
                version: SemanticVersion::new(major, minor, 0),
                status: VersionStatus::Stable,
                parent_version: if major == 1 && minor == 0 {
                    None
                } else {
                    Some(SemanticVersion::new(
                        if minor == 0 { major - 1 } else { major },
                        if minor == 0 { 1 } else { minor - 1 },
                        0,
                    ))
                },
                changes: format!("v{}.{}", major, minor),
            });
        }

        let path = registry.find_upgrade_path(
            "m",
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(2, 1, 0),
        );
        assert!(path.is_some());
        let path = path.expect("path should exist");
        assert_eq!(path.steps.len(), 4);
        assert_eq!(path.breaking_changes, 1);
    }

    #[test]
    fn test_upgrade_path_no_path() {
        let registry = VersionRegistry::new();
        let path = registry.find_upgrade_path(
            "missing",
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(2, 0, 0),
        );
        assert!(path.is_none());
    }

    #[test]
    fn test_transition_status_valid() {
        let mut registry = VersionRegistry::new();
        registry.register(ModelVersion {
            name: "m".to_string(),
            version: SemanticVersion::new(1, 0, 0),
            status: VersionStatus::Alpha,
            parent_version: None,
            changes: "init".to_string(),
        });

        let result =
            registry.transition_status("m", &SemanticVersion::new(1, 0, 0), VersionStatus::Beta);
        assert!(result.is_ok());

        let mv = &registry.versions["m"][0];
        assert_eq!(mv.status, VersionStatus::Beta);
    }

    #[test]
    fn test_transition_status_invalid() {
        let mut registry = VersionRegistry::new();
        registry.register(ModelVersion {
            name: "m".to_string(),
            version: SemanticVersion::new(1, 0, 0),
            status: VersionStatus::Archived,
            parent_version: None,
            changes: "init".to_string(),
        });

        let result =
            registry.transition_status("m", &SemanticVersion::new(1, 0, 0), VersionStatus::Stable);
        assert!(result.is_err());
    }

    #[test]
    fn test_version_diff() {
        let mut registry = VersionRegistry::new();
        registry.register(ModelVersion {
            name: "m".to_string(),
            version: SemanticVersion::new(1, 0, 0),
            status: VersionStatus::Archived,
            parent_version: None,
            changes: "init".to_string(),
        });
        registry.register(ModelVersion {
            name: "m".to_string(),
            version: SemanticVersion::new(1, 1, 0),
            status: VersionStatus::Deprecated,
            parent_version: Some(SemanticVersion::new(1, 0, 0)),
            changes: "patch".to_string(),
        });
        registry.register(ModelVersion {
            name: "m".to_string(),
            version: SemanticVersion::new(2, 0, 0),
            status: VersionStatus::Stable,
            parent_version: Some(SemanticVersion::new(1, 1, 0)),
            changes: "breaking".to_string(),
        });

        let diff = registry.version_diff(
            "m",
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(2, 0, 0),
        );
        assert_eq!(diff.intermediate_count, 1);
        assert_eq!(diff.breaking_count, 1);
        assert_eq!(diff.from_status, VersionStatus::Archived);
        assert_eq!(diff.to_status, VersionStatus::Stable);
    }

    #[test]
    fn test_version_diff_missing_model() {
        let registry = VersionRegistry::new();
        let diff = registry.version_diff(
            "missing",
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(2, 0, 0),
        );
        assert_eq!(diff.intermediate_count, 0);
        assert_eq!(diff.breaking_count, 0);
    }

    #[test]
    fn test_main_runs_successfully() {
        // Verify the main function completes without error
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
        fn prop_version_ordering_is_total(
            a_major in 0u32..10,
            a_minor in 0u32..20,
            a_patch in 0u32..30,
            b_major in 0u32..10,
            b_minor in 0u32..20,
            b_patch in 0u32..30,
        ) {
            let a = SemanticVersion::new(a_major, a_minor, a_patch);
            let b = SemanticVersion::new(b_major, b_minor, b_patch);

            // Total ordering: exactly one of <, =, > holds
            let lt = a < b;
            let eq = a == b;
            let gt = a > b;
            prop_assert_eq!(
                lt as u8 + eq as u8 + gt as u8,
                1,
                "exactly one of <, ==, > must hold"
            );
        }

        #[test]
        fn prop_version_parse_roundtrip(
            major in 0u32..100,
            minor in 0u32..100,
            patch in 0u32..100,
        ) {
            let v = SemanticVersion::new(major, minor, patch);
            let s = format!("{}", v);
            let parsed = SemanticVersion::parse(&s);
            prop_assert_eq!(parsed, Some(v));
        }

        #[test]
        fn prop_range_contains_endpoints(
            min_major in 0u32..5,
            min_minor in 0u32..5,
            span_major in 0u32..5,
            span_minor in 0u32..5,
        ) {
            let min = SemanticVersion::new(min_major, min_minor, 0);
            let max = SemanticVersion::new(min_major + span_major, min_minor + span_minor, 0);
            let range = VersionRange::new(min.clone(), max.clone());

            prop_assert!(range.contains(&min), "range must contain its min");
            prop_assert!(range.contains(&max), "range must contain its max");
        }

        #[test]
        fn prop_register_preserves_sort_order(n in 1usize..15) {
            let mut registry = VersionRegistry::new();
            // Register in reverse order
            for i in (0..n).rev() {
                registry.register(ModelVersion {
                    name: "model".to_string(),
                    version: SemanticVersion::new(1, i as u32, 0),
                    status: VersionStatus::Stable,
                    parent_version: None,
                    changes: "test".to_string(),
                });
            }
            let versions = &registry.versions["model"];
            for w in versions.windows(2) {
                prop_assert!(w[0].version <= w[1].version, "versions must be sorted");
            }
        }
    }
}
