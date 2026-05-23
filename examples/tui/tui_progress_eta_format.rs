//! # TUI Progress ETA Format
//!
//! Format a duration in seconds into human-readable form:
//!   <60: "Ns"
//!   <3600: "MmSSs"
//!   <86400: "HhMMm"
//!   ≥86400: "Dd HHh"
//!
//! Demonstrates the **TUI.26** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: indicatif duration formatting.
//!
//! Run with: cargo run --example tui_progress_eta_format
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EtaVerdict {
    Ok { formatted: String },
    InvalidInput,
}

pub fn format(seconds: f64) -> EtaVerdict {
    if !seconds.is_finite() || seconds < 0.0 {
        return EtaVerdict::InvalidInput;
    }
    let s = seconds as u64;
    let formatted = if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        let m = s / 60;
        let rem = s % 60;
        format!("{m}m{rem:02}s")
    } else if s < 86400 {
        let h = s / 3600;
        let rem = (s % 3600) / 60;
        format!("{h}h{rem:02}m")
    } else {
        let d = s / 86400;
        let rem = (s % 86400) / 3600;
        format!("{d}d {rem:02}h")
    };
    EtaVerdict::Ok { formatted }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_eta_format")?;

    println!("seconds: {:?}", format(45.0));
    println!("minutes: {:?}", format(125.0));
    println!("hours: {:?}", format(7200.0));
    println!("days: {:?}", format(180_000.0));
    println!("zero: {:?}", format(0.0));
    println!("invalid: {:?}", format(-1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_60_seconds() {
        let v = format(45.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "45s");
        }
    }

    #[test]
    fn minutes_format() {
        let v = format(125.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "2m05s");
        }
    }

    #[test]
    fn exactly_one_hour() {
        let v = format(3600.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "1h00m");
        }
    }

    #[test]
    fn hours_format() {
        let v = format(7200.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "2h00m");
        }
    }

    #[test]
    fn days_format() {
        let v = format(180_000.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "2d 02h");
        }
    }

    #[test]
    fn zero_seconds() {
        let v = format(0.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "0s");
        }
    }

    #[test]
    fn negative_invalid() {
        assert_eq!(format(-1.0), EtaVerdict::InvalidInput);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(format(f64::NAN), EtaVerdict::InvalidInput);
    }

    #[test]
    fn infinity_invalid() {
        assert_eq!(format(f64::INFINITY), EtaVerdict::InvalidInput);
    }

    #[test]
    fn boundary_at_60() {
        let v = format(60.0);
        if let EtaVerdict::Ok { formatted } = v {
            assert_eq!(formatted, "1m00s");
        }
    }

    #[test]
    fn deterministic() {
        let a = format(7200.0);
        let b = format(7200.0);
        assert_eq!(a, b);
    }
}
