//! # Contracts-Macros YAML Envelope Size Audit
//!
//! Flag YAML files exceeding configured size limits (lines + bytes).
//! Returns each file's status (Ok / TooManyLines / TooManyBytes).
//!
//! Demonstrates the **CMM.80** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub repo file size limits (100 MB hard, 50 MB warn);
//!  Linus Torvalds, "small files keep you sane".
//!
//! Run with: cargo run --example contracts_macros_yaml_envelope_size_audit
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum FileStatus {
    Ok,
    TooManyLines,
    TooManyBytes,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok { per_file: Vec<(String, FileStatus)> },
    InvalidConfig,
}

pub fn audit(files: &[(&str, u32, u32)], max_lines: u32, max_bytes: u32) -> AuditVerdict {
    if files.is_empty() || max_lines == 0 || max_bytes == 0 {
        return AuditVerdict::InvalidConfig;
    }
    let mut per_file: Vec<(String, FileStatus)> = Vec::with_capacity(files.len());
    for (name, lines, bytes) in files {
        let status = if *lines > max_lines {
            FileStatus::TooManyLines
        } else if *bytes > max_bytes {
            FileStatus::TooManyBytes
        } else {
            FileStatus::Ok
        };
        per_file.push(((*name).to_string(), status));
    }
    AuditVerdict::Ok { per_file }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_envelope_size_audit")?;

    let files = [
        ("ok.yaml", 50, 1500),
        ("too_long.yaml", 600, 30_000),
        ("dense.yaml", 100, 60_000),
    ];
    println!("audit: {:?}", audit(&files, 500, 50_000));
    println!("invalid: {:?}", audit(&[], 500, 50_000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_file_ok() {
        let files = [("a.yaml", 50, 1500)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::Ok);
        }
    }

    #[test]
    fn too_many_lines_flagged() {
        let files = [("a.yaml", 1000, 100)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::TooManyLines);
        }
    }

    #[test]
    fn too_many_bytes_flagged() {
        let files = [("a.yaml", 50, 100_000)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::TooManyBytes);
        }
    }

    #[test]
    fn lines_take_precedence_over_bytes() {
        let files = [("a.yaml", 1000, 100_000)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::TooManyLines);
        }
    }

    #[test]
    fn empty_files_rejected() {
        assert_eq!(audit(&[], 500, 50_000), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_lines_rejected() {
        let files = [("a.yaml", 50, 100)];
        assert_eq!(audit(&files, 0, 50_000), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_bytes_rejected() {
        let files = [("a.yaml", 50, 100)];
        assert_eq!(audit(&files, 500, 0), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_limit_ok() {
        let files = [("a.yaml", 500, 50_000)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::Ok);
        }
    }

    #[test]
    fn one_over_limit_flagged() {
        let files = [("a.yaml", 501, 100)];
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].1, FileStatus::TooManyLines);
        }
    }

    #[test]
    fn deterministic() {
        let files = [("a.yaml", 50, 100)];
        let r1 = audit(&files, 500, 50_000);
        let r2 = audit(&files, 500, 50_000);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_files_handled() {
        let files: Vec<(&str, u32, u32)> = (0..20).map(|_| ("f.yaml", 50, 100)).collect();
        let v = audit(&files, 500, 50_000);
        if let AuditVerdict::Ok { per_file } = v {
            assert_eq!(per_file.len(), 20);
            assert!(per_file.iter().all(|(_, s)| *s == FileStatus::Ok));
        }
    }
}
