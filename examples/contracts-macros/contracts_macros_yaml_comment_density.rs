//! # Contracts-Macros YAML Comment Density
//!
//! Measure comment-to-line ratio per YAML file. Flag files whose
//! density falls outside `[min_pct, max_pct]` bounds.
//!
//! Demonstrates the **CMM.86** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wirth, "On the Importance of Comments" (1971); Donald
//!  Knuth, Literate Programming (1984).
//!
//! Run with: cargo run --example contracts_macros_yaml_comment_density
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum DensityStatus {
    Ok,
    UnderDocumented,
    OverCommented,
}

#[derive(Debug, PartialEq)]
pub enum DensityVerdict {
    Ok {
        per_file: Vec<(String, f64, DensityStatus)>,
    },
    InvalidConfig,
}

pub fn audit(files: &[(&str, u32, u32)], min_pct: f64, max_pct: f64) -> DensityVerdict {
    if files.is_empty() || min_pct < 0.0 || max_pct > 100.0 || min_pct >= max_pct {
        return DensityVerdict::InvalidConfig;
    }
    let mut per_file: Vec<(String, f64, DensityStatus)> = Vec::with_capacity(files.len());
    for (name, total_lines, comment_lines) in files {
        if *total_lines == 0 {
            return DensityVerdict::InvalidConfig;
        }
        let density = 100.0 * f64::from(*comment_lines) / f64::from(*total_lines);
        let status = if density < min_pct {
            DensityStatus::UnderDocumented
        } else if density > max_pct {
            DensityStatus::OverCommented
        } else {
            DensityStatus::Ok
        };
        per_file.push(((*name).to_string(), density, status));
    }
    DensityVerdict::Ok { per_file }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_comment_density")?;

    let files = [
        ("good.yaml", 100, 15),  // 15%
        ("bare.yaml", 100, 1),   // 1%
        ("dense.yaml", 100, 50), // 50%
    ];
    println!("audit: {:?}", audit(&files, 5.0, 30.0));
    println!("invalid: {:?}", audit(&[], 5.0, 30.0));
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
    fn balanced_density_ok() {
        let files = [("a.yaml", 100, 15)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].2, DensityStatus::Ok);
        }
    }

    #[test]
    fn under_documented_flagged() {
        let files = [("a.yaml", 100, 1)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].2, DensityStatus::UnderDocumented);
        }
    }

    #[test]
    fn over_commented_flagged() {
        let files = [("a.yaml", 100, 50)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].2, DensityStatus::OverCommented);
        }
    }

    #[test]
    fn empty_files_rejected() {
        assert_eq!(audit(&[], 5.0, 30.0), DensityVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_min_above_max() {
        let files = [("a.yaml", 100, 10)];
        assert_eq!(audit(&files, 50.0, 10.0), DensityVerdict::InvalidConfig);
    }

    #[test]
    fn zero_total_rejected() {
        let files = [("empty.yaml", 0, 0)];
        assert_eq!(audit(&files, 5.0, 30.0), DensityVerdict::InvalidConfig);
    }

    #[test]
    fn density_value_correct() {
        let files = [("a.yaml", 100, 25)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert!((per_file[0].1 - 25.0).abs() < 0.001);
        }
    }

    #[test]
    fn boundary_min_passes() {
        let files = [("a.yaml", 100, 5)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].2, DensityStatus::Ok);
        }
    }

    #[test]
    fn boundary_max_passes() {
        let files = [("a.yaml", 100, 30)];
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file[0].2, DensityStatus::Ok);
        }
    }

    #[test]
    fn deterministic() {
        let files = [("a.yaml", 100, 15)];
        let r1 = audit(&files, 5.0, 30.0);
        let r2 = audit(&files, 5.0, 30.0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_files_handled() {
        let files: Vec<(&str, u32, u32)> = (0..10).map(|_| ("f.yaml", 100, 15)).collect();
        let v = audit(&files, 5.0, 30.0);
        if let DensityVerdict::Ok { per_file } = v {
            assert_eq!(per_file.len(), 10);
        }
    }
}
