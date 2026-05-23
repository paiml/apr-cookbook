//! # Monitoring Aggregation Window Picker
//!
//! Window types for metric streams:
//!   Tumbling: non-overlapping fixed [start, end). Easiest to reason about.
//!   Sliding: continuous [t-W, t]. Smooth but expensive.
//!   Hopping: fixed slide between windows; trade-off.
//!
//! Picker:
//!   want_smooth + low_qps → Sliding
//!   want_simple → Tumbling
//!   high_qps + smooth → Hopping
//!
//! Demonstrates the **MON.31** recipe for PMAT-151 (monitoring round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Apache Flink window types specification.
//!
//! Run with: cargo run --example monitor_aggregation_window
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    Tumbling { secs: u32 },
    Sliding { secs: u32 },
    Hopping { window_secs: u32, slide_secs: u32 },
}

#[derive(Debug, PartialEq)]
pub enum WindowVerdict {
    Ok {
        kind: WindowKind,
        memory_overhead_factor: f64,
    },
    InvalidQps,
}

const HIGH_QPS: u32 = 10_000;

pub fn pick(qps: u32, want_smooth: bool, simplicity_priority: bool) -> WindowVerdict {
    if qps == 0 {
        return WindowVerdict::InvalidQps;
    }
    let kind = if simplicity_priority {
        WindowKind::Tumbling { secs: 60 }
    } else if want_smooth && qps < HIGH_QPS {
        WindowKind::Sliding { secs: 60 }
    } else if want_smooth {
        WindowKind::Hopping {
            window_secs: 60,
            slide_secs: 10,
        }
    } else {
        WindowKind::Tumbling { secs: 60 }
    };
    let memory_overhead_factor = match kind {
        WindowKind::Tumbling { .. } => 1.0,
        WindowKind::Sliding { .. } => 5.0,
        WindowKind::Hopping { .. } => 6.0,
    };
    WindowVerdict::Ok {
        kind,
        memory_overhead_factor,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_aggregation_window")?;

    println!("low qps smooth: {:?}", pick(100, true, false));
    println!("high qps smooth: {:?}", pick(50_000, true, false));
    println!("simple: {:?}", pick(1000, true, true));
    println!("invalid: {:?}", pick(0, true, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_qps_smooth_picks_sliding() {
        let v = pick(100, true, false);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Sliding { .. }));
        }
    }

    #[test]
    fn high_qps_smooth_picks_hopping() {
        let v = pick(50_000, true, false);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Hopping { .. }));
        }
    }

    #[test]
    fn simplicity_picks_tumbling() {
        let v = pick(100, true, true);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Tumbling { .. }));
        }
    }

    #[test]
    fn no_smooth_picks_tumbling() {
        let v = pick(100, false, false);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Tumbling { .. }));
        }
    }

    #[test]
    fn invalid_zero_qps() {
        assert_eq!(pick(0, true, false), WindowVerdict::InvalidQps);
    }

    #[test]
    fn tumbling_lowest_overhead() {
        let tumb = pick(100, false, false);
        let sliding = pick(100, true, false);
        if let (
            WindowVerdict::Ok {
                memory_overhead_factor: t,
                ..
            },
            WindowVerdict::Ok {
                memory_overhead_factor: s,
                ..
            },
        ) = (tumb, sliding)
        {
            assert!(s > t);
        }
    }

    #[test]
    fn boundary_at_high_qps() {
        // exactly HIGH_QPS → still sliding (rule is `< HIGH_QPS`).
        let v = pick(HIGH_QPS, true, false);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Hopping { .. }));
        }
    }

    #[test]
    fn just_below_high_qps_sliding() {
        let v = pick(HIGH_QPS - 1, true, false);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Sliding { .. }));
        }
    }

    #[test]
    fn simplicity_overrides_smooth() {
        let v = pick(100, true, true);
        if let WindowVerdict::Ok { kind, .. } = v {
            assert!(matches!(kind, WindowKind::Tumbling { .. }));
        }
    }

    #[test]
    fn hopping_has_window_and_slide() {
        let v = pick(50_000, true, false);
        if let WindowVerdict::Ok {
            kind:
                WindowKind::Hopping {
                    window_secs,
                    slide_secs,
                },
            ..
        } = v
        {
            assert!(window_secs > slide_secs);
        }
    }
}
