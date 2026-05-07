//! # Contracts-Macros Coverage Band Check
//!
//! Verify each module's coverage falls within the configured band
//! `[min_pct, max_pct]`. Below band → undertested. Above band →
//! suspicious (likely spurious tests). Returns sorted offenders and
//! the count in-band.
//!
//! Demonstrates the **CMM.139** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SQALE quality model §5.2; ISO/IEC 25010 reliability bands.
//!
//! Run with: cargo run --example contracts_macros_coverage_band_check
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BandVerdict {
    Ok {
        offenders: Vec<String>,
        in_band_count: u32,
    },
    InvalidConfig,
}

pub fn check(modules: &[(&str, u32)], min_pct: u32, max_pct: u32) -> BandVerdict {
    if modules.is_empty() || max_pct > 100 || min_pct > max_pct {
        return BandVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = modules
        .iter()
        .filter(|(_, pct)| *pct < min_pct || *pct > max_pct)
        .map(|(name, _)| (*name).to_string())
        .collect();
    offenders.sort();
    let in_band = (modules.len() as u32) - (offenders.len() as u32);
    BandVerdict::Ok {
        offenders,
        in_band_count: in_band,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_coverage_band_check")?;

    let modules = [("auth", 96), ("api", 70), ("util", 99)];
    println!("band 95-100: {:?}", check(&modules, 95, 100));
    println!("invalid: {:?}", check(&[], 95, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn in_band_module_no_offender() {
        let v = check(&[("a", 96)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn under_band_offender() {
        let v = check(&[("a", 70)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["a".to_string()]);
        }
    }

    #[test]
    fn at_min_boundary_in_band() {
        let v = check(&[("a", 95)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn at_max_boundary_in_band() {
        let v = check(&[("a", 100)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], 95, 100), BandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_max_above_100() {
        assert_eq!(check(&[("a", 50)], 0, 101), BandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_inverted_band() {
        assert_eq!(check(&[("a", 50)], 100, 50), BandVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", 50)], 0, 100);
        let r2 = check(&[("a", 50)], 0, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn in_band_count_correct() {
        let v = check(&[("a", 96), ("b", 70), ("c", 99)], 95, 100);
        if let BandVerdict::Ok { in_band_count, .. } = v {
            assert_eq!(in_band_count, 2);
        }
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", 50), ("alpha", 50)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_modules_handled() {
        let modules: Vec<(&str, u32)> = (0..30).map(|_| ("m", 50)).collect();
        let v = check(&modules, 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders.len(), 30);
        }
    }

    #[test]
    fn unicode_module_supported() {
        let v = check(&[("café", 50)], 95, 100);
        if let BandVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["café".to_string()]);
        }
    }
}
