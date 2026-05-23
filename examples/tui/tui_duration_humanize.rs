//! # TUI Duration Humanize
//!
//! Convert seconds to compact human-readable string like
//! `"2d 3h 4m 5s"`, dropping leading zero units. Used in TUIs for
//! relative timestamps.
//!
//! Demonstrates the **TUI.68** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU coreutils `time` output; humanize.py library.
//!
//! Run with: cargo run --example tui_duration_humanize
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DurationVerdict {
    Ok { rendered: String },
    InvalidConfig,
}

pub fn humanize(seconds: u64) -> DurationVerdict {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    let parts: Vec<String> = [(days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")]
        .iter()
        .filter(|(v, _)| *v > 0)
        .map(|(v, suffix)| format!("{v}{suffix}"))
        .collect();
    let rendered = if parts.is_empty() {
        "0s".to_string()
    } else {
        parts.join(" ")
    };
    DurationVerdict::Ok { rendered }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_duration_humanize")?;

    println!("0: {:?}", humanize(0));
    println!("65: {:?}", humanize(65));
    println!("90061: {:?}", humanize(90061));
    println!("max: {:?}", humanize(u64::MAX));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_is_zero_seconds() {
        let v = humanize(0);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "0s");
        }
    }

    #[test]
    fn under_minute_seconds_only() {
        let v = humanize(45);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "45s");
        }
    }

    #[test]
    fn over_minute_combines() {
        let v = humanize(65);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1m 5s");
        }
    }

    #[test]
    fn over_hour_combines() {
        let v = humanize(3661);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1h 1m 1s");
        }
    }

    #[test]
    fn over_day_combines() {
        // 1 day + 1 hour + 1 min + 1 sec = 86400 + 3600 + 60 + 1.
        let v = humanize(90061);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1d 1h 1m 1s");
        }
    }

    #[test]
    fn drops_zero_middle_units() {
        // 1 day + 0 hour + 1 min + 0 sec = 86400 + 60.
        let v = humanize(86460);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1d 1m");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = humanize(123);
        let r2 = humanize(123);
        assert_eq!(r1, r2);
    }

    #[test]
    fn one_minute_no_seconds() {
        let v = humanize(60);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1m");
        }
    }

    #[test]
    fn one_hour_no_minutes() {
        let v = humanize(3600);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1h");
        }
    }

    #[test]
    fn one_day_no_hours() {
        let v = humanize(86_400);
        if let DurationVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "1d");
        }
    }

    #[test]
    fn very_large_value_works() {
        let v = humanize(u64::MAX);
        assert!(matches!(v, DurationVerdict::Ok { .. }));
    }

    #[test]
    fn rendered_contains_at_most_four_parts() {
        let v = humanize(90061);
        if let DurationVerdict::Ok { rendered } = v {
            let parts: Vec<&str> = rendered.split(' ').collect();
            assert!(parts.len() <= 4);
        }
    }
}
