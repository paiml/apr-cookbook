//! # Contracts-Macros Invariant Priority Band
//!
//! Bucket invariants into Low/Medium/High/Critical priority bands.
//! Returns counts per band and the highest-priority band that has
//! any items.
//!
//! Demonstrates the **CMM.192** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PagerDuty severity bands; ITIL urgency-impact matrix.
//!
//! Run with: cargo run --example contracts_macros_invariant_priority_band
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PriorityBand {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum BandVerdict {
    Ok {
        low: u32,
        medium: u32,
        high: u32,
        critical: u32,
        peak_band: PriorityBand,
    },
    InvalidConfig,
}

pub fn classify(priorities: &[u8]) -> BandVerdict {
    if priorities.is_empty() {
        return BandVerdict::InvalidConfig;
    }
    for p in priorities {
        if !(1..=10).contains(p) {
            return BandVerdict::InvalidConfig;
        }
    }
    let mut low = 0u32;
    let mut medium = 0u32;
    let mut high = 0u32;
    let mut critical = 0u32;
    for p in priorities {
        match *p {
            1..=3 => low += 1,
            4..=6 => medium += 1,
            7..=8 => high += 1,
            _ => critical += 1,
        }
    }
    let peak = if critical > 0 {
        PriorityBand::Critical
    } else if high > 0 {
        PriorityBand::High
    } else if medium > 0 {
        PriorityBand::Medium
    } else if low > 0 {
        PriorityBand::Low
    } else {
        PriorityBand::None
    };
    BandVerdict::Ok {
        low,
        medium,
        high,
        critical,
        peak_band: peak,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_priority_band")?;

    println!("mixed: {:?}", classify(&[1, 5, 7, 10]));
    println!("invalid: {:?}", classify(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(classify(&[]), BandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_priority() {
        assert_eq!(classify(&[0]), BandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_over_ten() {
        assert_eq!(classify(&[11]), BandVerdict::InvalidConfig);
    }

    #[test]
    fn low_band_correct() {
        let v = classify(&[1, 2, 3]);
        if let BandVerdict::Ok { low, .. } = v {
            assert_eq!(low, 3);
        }
    }

    #[test]
    fn medium_band_correct() {
        let v = classify(&[4, 5, 6]);
        if let BandVerdict::Ok { medium, .. } = v {
            assert_eq!(medium, 3);
        }
    }

    #[test]
    fn high_band_correct() {
        let v = classify(&[7, 8]);
        if let BandVerdict::Ok { high, .. } = v {
            assert_eq!(high, 2);
        }
    }

    #[test]
    fn critical_band_correct() {
        let v = classify(&[9, 10]);
        if let BandVerdict::Ok { critical, .. } = v {
            assert_eq!(critical, 2);
        }
    }

    #[test]
    fn peak_band_critical_when_any() {
        let v = classify(&[1, 5, 10]);
        if let BandVerdict::Ok { peak_band, .. } = v {
            assert_eq!(peak_band, PriorityBand::Critical);
        }
    }

    #[test]
    fn peak_band_high_when_no_critical() {
        let v = classify(&[1, 7]);
        if let BandVerdict::Ok { peak_band, .. } = v {
            assert_eq!(peak_band, PriorityBand::High);
        }
    }

    #[test]
    fn peak_band_low_when_only_low() {
        let v = classify(&[1, 2, 3]);
        if let BandVerdict::Ok { peak_band, .. } = v {
            assert_eq!(peak_band, PriorityBand::Low);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = classify(&[1, 5]);
        let r2 = classify(&[1, 5]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_3_low() {
        let v = classify(&[3]);
        if let BandVerdict::Ok { low, .. } = v {
            assert_eq!(low, 1);
        }
    }

    #[test]
    fn boundary_4_medium() {
        let v = classify(&[4]);
        if let BandVerdict::Ok { medium, .. } = v {
            assert_eq!(medium, 1);
        }
    }

    #[test]
    fn boundary_8_high() {
        let v = classify(&[8]);
        if let BandVerdict::Ok { high, .. } = v {
            assert_eq!(high, 1);
        }
    }
}
