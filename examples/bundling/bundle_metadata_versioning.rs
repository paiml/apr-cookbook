//! # Bundle Metadata Versioning Policy
//!
//! Bundle metadata follows semver (major.minor.patch). Reader compat:
//!
//! - same major: read OK (forward-compat for added fields)
//! - reader.major > bundle.major: read OK (deprecation grace)
//! - reader.major < bundle.major: REJECT (would silently lose fields)
//!
//! This recipe builds the picker.
//!
//! Demonstrates the **BUNDLE.18** recipe for PMAT-136 (bundling round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SemVer 2.0 specification.
//!
//! Run with: cargo run --example bundle_metadata_versioning
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Compatible,
    NewerMajorBundle { reader: u32, bundle: u32 },
    Forwards { dropped_fields_possible: bool },
}

pub fn check(reader: Version, bundle: Version) -> CompatVerdict {
    if reader.major < bundle.major {
        return CompatVerdict::NewerMajorBundle {
            reader: reader.major,
            bundle: bundle.major,
        };
    }
    if reader.major > bundle.major {
        return CompatVerdict::Forwards {
            dropped_fields_possible: false,
        };
    }
    if reader.minor < bundle.minor {
        return CompatVerdict::Forwards {
            dropped_fields_possible: true,
        };
    }
    CompatVerdict::Compatible
}

pub fn parse(s: &str) -> Option<Version> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some(Version {
        major: parts[0].parse().ok()?,
        minor: parts[1].parse().ok()?,
        patch: parts[2].parse().ok()?,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_metadata_versioning")?;

    let pairs = [
        ("2.0.0", "2.0.0"),
        ("2.1.0", "2.0.0"),
        ("2.0.0", "2.1.0"),
        ("3.0.0", "2.0.0"),
        ("1.0.0", "2.0.0"),
    ];
    for (r, b) in pairs {
        let r = parse(r).unwrap();
        let b = parse(b).unwrap();
        println!("{r:?} reading {b:?}: {:?}", check(r, b));
    }
    println!("invalid parse: {:?}", parse("not-a-version"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        parse(s).expect("test version literal must parse")
    }

    #[test]
    fn versioning_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_version_compatible() {
        assert_eq!(check(v("2.0.0"), v("2.0.0")), CompatVerdict::Compatible);
    }

    #[test]
    fn newer_minor_reader_compatible() {
        // Reader 2.1 reading bundle 2.0 → can read all known fields.
        assert_eq!(check(v("2.1.0"), v("2.0.0")), CompatVerdict::Compatible);
    }

    #[test]
    fn older_minor_reader_forwards_compat_flagged() {
        // Reader 2.0 reading bundle 2.1 → may have unknown fields.
        let v = check(v("2.0.0"), v("2.1.0"));
        assert_eq!(
            v,
            CompatVerdict::Forwards {
                dropped_fields_possible: true
            }
        );
    }

    #[test]
    fn newer_major_reader_forwards_compat() {
        // Reader 3.0 reading bundle 2.0 → deprecation grace, no field loss expected.
        let v = check(v("3.0.0"), v("2.0.0"));
        assert_eq!(
            v,
            CompatVerdict::Forwards {
                dropped_fields_possible: false
            }
        );
    }

    #[test]
    fn older_major_reader_rejected() {
        // Reader 1.0 reading bundle 2.0 → SILENT FIELD LOSS, must reject.
        let v = check(v("1.0.0"), v("2.0.0"));
        assert!(matches!(v, CompatVerdict::NewerMajorBundle { .. }));
    }

    #[test]
    fn parse_typical() {
        let v = parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn parse_invalid_too_many_parts() {
        assert!(parse("1.2.3.4").is_none());
    }

    #[test]
    fn parse_invalid_too_few_parts() {
        assert!(parse("1.2").is_none());
    }

    #[test]
    fn parse_invalid_non_numeric() {
        assert!(parse("a.b.c").is_none());
    }

    #[test]
    fn patch_does_not_affect_compat() {
        // Patch differences are always compatible (per semver).
        assert_eq!(check(v("2.0.5"), v("2.0.0")), CompatVerdict::Compatible);
        assert_eq!(check(v("2.0.0"), v("2.0.5")), CompatVerdict::Compatible);
    }
}
