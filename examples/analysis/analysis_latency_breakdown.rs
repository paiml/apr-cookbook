//! # Analysis Latency Breakdown Renderer
//!
//! End-to-end latency = sum of per-stage durations. Renders a textual
//! stack-bar showing each stage as a fraction. Stages are ordered by
//! contribution descending. This recipe builds the renderer + the
//! "biggest contributor" classifier.
//!
//! Demonstrates the **ANL.54** recipe for PMAT-131 (analysis coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gregg, B. (2016). Systems Performance §6.3.
//!
//! Run with: cargo run --example analysis_latency_breakdown
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone)]
pub struct Stage {
    pub name: String,
    pub duration_ms: f64,
}

#[derive(Debug, PartialEq)]
pub enum BreakdownVerdict {
    Ok {
        rendered: String,
        biggest_pct: f64,
        biggest_name: String,
    },
    EmptyStages,
    InvalidDuration {
        stage: String,
    },
    AllZeroDuration,
}

pub fn render(stages: &[Stage]) -> BreakdownVerdict {
    if stages.is_empty() {
        return BreakdownVerdict::EmptyStages;
    }
    for s in stages {
        if !s.duration_ms.is_finite() || s.duration_ms < 0.0 {
            return BreakdownVerdict::InvalidDuration {
                stage: s.name.clone(),
            };
        }
    }
    let total: f64 = stages.iter().map(|s| s.duration_ms).sum();
    if total == 0.0 {
        return BreakdownVerdict::AllZeroDuration;
    }
    let mut sorted: Vec<&Stage> = stages.iter().collect();
    sorted.sort_by(|a, b| {
        b.duration_ms
            .partial_cmp(&a.duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = String::new();
    for s in &sorted {
        let pct = s.duration_ms / total * 100.0;
        out.push_str(&format!(
            "{:<20} {:>6.2} ms  {:>5.1}%\n",
            truncate(&s.name, 20),
            s.duration_ms,
            pct
        ));
    }
    let biggest = sorted[0];
    let biggest_pct = biggest.duration_ms / total * 100.0;
    BreakdownVerdict::Ok {
        rendered: out,
        biggest_pct,
        biggest_name: biggest.name.clone(),
    }
}

fn truncate(s: &str, width: usize) -> String {
    if s.len() <= width {
        s.to_string()
    } else {
        format!("{}…", &s[..width.saturating_sub(1)])
    }
}

#[derive(Debug, PartialEq)]
pub enum BottleneckTier {
    Balanced,
    SingleStageDominant { pct: f64 },
    TwoStagesDominant,
}

pub fn classify_bottleneck(stages: &[Stage]) -> Option<BottleneckTier> {
    if stages.is_empty() {
        return None;
    }
    let total: f64 = stages.iter().map(|s| s.duration_ms).sum();
    if total <= 0.0 {
        return None;
    }
    let mut pcts: Vec<f64> = stages
        .iter()
        .map(|s| s.duration_ms / total * 100.0)
        .collect();
    pcts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = pcts[0];
    let top_two = pcts.iter().take(2).sum::<f64>();
    Some(if top >= 60.0 {
        BottleneckTier::SingleStageDominant { pct: top }
    } else if top_two >= 80.0 {
        BottleneckTier::TwoStagesDominant
    } else {
        BottleneckTier::Balanced
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_latency_breakdown")?;

    let stages = vec![
        Stage {
            name: "tokenize".into(),
            duration_ms: 5.0,
        },
        Stage {
            name: "embed".into(),
            duration_ms: 12.0,
        },
        Stage {
            name: "decode".into(),
            duration_ms: 80.0,
        },
        Stage {
            name: "detok".into(),
            duration_ms: 3.0,
        },
    ];
    println!("{:?}", render(&stages));
    println!("classify: {:?}", classify_bottleneck(&stages));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Stage> {
        vec![
            Stage {
                name: "a".into(),
                duration_ms: 10.0,
            },
            Stage {
                name: "b".into(),
                duration_ms: 30.0,
            },
            Stage {
                name: "c".into(),
                duration_ms: 60.0,
            },
        ]
    }

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_render_includes_all_stages() {
        if let BreakdownVerdict::Ok { rendered, .. } = render(&sample()) {
            assert!(rendered.contains("a"));
            assert!(rendered.contains("b"));
            assert!(rendered.contains("c"));
        }
    }

    #[test]
    fn biggest_stage_correctly_identified() {
        if let BreakdownVerdict::Ok {
            biggest_name,
            biggest_pct,
            ..
        } = render(&sample())
        {
            assert_eq!(biggest_name, "c");
            assert!((biggest_pct - 60.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(render(&[]), BreakdownVerdict::EmptyStages);
    }

    #[test]
    fn negative_duration_rejected() {
        let bad = vec![Stage {
            name: "x".into(),
            duration_ms: -1.0,
        }];
        let v = render(&bad);
        assert!(matches!(v, BreakdownVerdict::InvalidDuration { .. }));
    }

    #[test]
    fn nan_duration_rejected() {
        let bad = vec![Stage {
            name: "x".into(),
            duration_ms: f64::NAN,
        }];
        let v = render(&bad);
        assert!(matches!(v, BreakdownVerdict::InvalidDuration { .. }));
    }

    #[test]
    fn all_zero_durations_rejected() {
        let zeros = vec![
            Stage {
                name: "a".into(),
                duration_ms: 0.0,
            },
            Stage {
                name: "b".into(),
                duration_ms: 0.0,
            },
        ];
        assert_eq!(render(&zeros), BreakdownVerdict::AllZeroDuration);
    }

    #[test]
    fn bottleneck_single_stage_dominant() {
        // One stage at 80%, others small.
        let s = vec![
            Stage {
                name: "a".into(),
                duration_ms: 5.0,
            },
            Stage {
                name: "b".into(),
                duration_ms: 5.0,
            },
            Stage {
                name: "big".into(),
                duration_ms: 90.0,
            },
        ];
        let t = classify_bottleneck(&s).unwrap();
        assert!(matches!(t, BottleneckTier::SingleStageDominant { .. }));
    }

    #[test]
    fn bottleneck_balanced() {
        // 3 equal stages.
        let s = vec![
            Stage {
                name: "a".into(),
                duration_ms: 100.0,
            },
            Stage {
                name: "b".into(),
                duration_ms: 100.0,
            },
            Stage {
                name: "c".into(),
                duration_ms: 100.0,
            },
        ];
        let t = classify_bottleneck(&s).unwrap();
        assert_eq!(t, BottleneckTier::Balanced);
    }

    #[test]
    fn bottleneck_two_stages_dominant() {
        // 50% + 35% top two = 85%.
        let s = vec![
            Stage {
                name: "a".into(),
                duration_ms: 50.0,
            },
            Stage {
                name: "b".into(),
                duration_ms: 35.0,
            },
            Stage {
                name: "c".into(),
                duration_ms: 15.0,
            },
        ];
        let t = classify_bottleneck(&s).unwrap();
        assert_eq!(t, BottleneckTier::TwoStagesDominant);
    }

    #[test]
    fn bottleneck_classify_empty_returns_none() {
        assert!(classify_bottleneck(&[]).is_none());
    }
}
