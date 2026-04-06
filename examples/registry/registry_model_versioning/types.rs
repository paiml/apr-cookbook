#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

// --- Core Types ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}
impl SemanticVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
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
    pub fn is_breaking_from(&self, other: &Self) -> bool {
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
pub enum VersionStatus {
    Alpha,
    Beta,
    Stable,
    Deprecated,
    Archived,
}
impl VersionStatus {
    pub fn can_transition_to(&self, target: &Self) -> bool {
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
pub struct ModelVersion {
    pub name: String,
    pub version: SemanticVersion,
    pub status: VersionStatus,
    pub parent_version: Option<SemanticVersion>,
    pub changes: String,
}

#[derive(Debug, Clone)]
pub struct VersionRange {
    pub min: SemanticVersion,
    pub max: SemanticVersion,
}
impl VersionRange {
    pub const fn new(min: SemanticVersion, max: SemanticVersion) -> Self {
        Self { min, max }
    }
    pub fn contains(&self, v: &SemanticVersion) -> bool {
        v >= &self.min && v <= &self.max
    }
}

#[derive(Debug, Clone)]
pub struct CompatibilityRule {
    pub source_model: String,
    pub source_range: VersionRange,
    pub target_model: String,
    pub target_range: VersionRange,
    pub compatible: bool,
}

#[derive(Debug, Clone)]
pub struct UpgradePath {
    pub steps: Vec<SemanticVersion>,
    pub breaking_changes: usize,
    pub migration_notes: Vec<String>,
}

#[derive(Debug)]
pub struct VersionDiff {
    pub intermediate_count: usize,
    pub breaking_count: usize,
    pub from_status: VersionStatus,
    pub to_status: VersionStatus,
}

#[derive(Debug, Clone)]
pub struct DependencyRule {
    pub model: String,
    pub version: SemanticVersion,
    pub depends_on_model: String,
    pub depends_on_range: VersionRange,
}

#[derive(Debug)]
pub struct VersionRegistry {
    pub versions: HashMap<String, Vec<ModelVersion>>,
    pub compatibility_rules: Vec<CompatibilityRule>,
    pub dependencies: Vec<DependencyRule>,
}
impl VersionRegistry {
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            compatibility_rules: Vec::new(),
            dependencies: Vec::new(),
        }
    }
    pub fn register(&mut self, mv: ModelVersion) {
        let e = self.versions.entry(mv.name.clone()).or_default();
        e.push(mv);
        e.sort_by(|a, b| a.version.cmp(&b.version));
    }
    pub fn add_compatibility_rule(&mut self, rule: CompatibilityRule) {
        self.compatibility_rules.push(rule);
    }
    pub fn add_dependency(
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
    pub fn check_compatibility(
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
    pub fn resolve_dependencies(
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
    pub fn find_upgrade_path(
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
    pub fn transition_status(
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
    pub fn version_diff(
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
