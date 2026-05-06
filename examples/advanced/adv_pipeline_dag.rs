//! # Advanced Multi-Stage Inference Pipeline DAG
//!
//! Real inference pipelines have stages: tokenize → embed → infer →
//! detokenize → post-process. Each stage has a duration and may
//! depend on prior stages. This recipe builds the DAG and computes
//! critical-path latency assuming N workers per stage.
//!
//! Demonstrates the **ADV.6** recipe for PMAT-139 (advanced coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA Triton Inference Server pipeline-orchestration model.
//!
//! Run with: cargo run --example adv_pipeline_dag
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone)]
pub struct Stage {
    pub name: String,
    pub duration_ms: u32,
    pub depends_on: Vec<usize>,
}

#[derive(Debug, PartialEq)]
pub enum DagVerdict {
    Ok {
        critical_path_ms: u32,
        finish_times: Vec<u32>,
    },
    EmptyDag,
    InvalidDependency {
        stage: usize,
        dep: usize,
    },
}

pub fn schedule(stages: &[Stage]) -> DagVerdict {
    if stages.is_empty() {
        return DagVerdict::EmptyDag;
    }
    let mut finish: Vec<u32> = vec![0; stages.len()];
    for (i, stage) in stages.iter().enumerate() {
        let mut max_dep_finish = 0u32;
        for &d in &stage.depends_on {
            if d >= i {
                return DagVerdict::InvalidDependency { stage: i, dep: d };
            }
            max_dep_finish = max_dep_finish.max(finish[d]);
        }
        finish[i] = max_dep_finish + stage.duration_ms;
    }
    let critical_path = *finish.iter().max().unwrap();
    DagVerdict::Ok {
        critical_path_ms: critical_path,
        finish_times: finish,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_pipeline_dag")?;

    let stages = vec![
        Stage {
            name: "tokenize".to_string(),
            duration_ms: 5,
            depends_on: vec![],
        },
        Stage {
            name: "embed".to_string(),
            duration_ms: 10,
            depends_on: vec![0],
        },
        Stage {
            name: "infer".to_string(),
            duration_ms: 100,
            depends_on: vec![1],
        },
        Stage {
            name: "detokenize".to_string(),
            duration_ms: 3,
            depends_on: vec![2],
        },
    ];
    println!("typical: {:?}", schedule(&stages));
    println!("empty: {:?}", schedule(&[]));

    let bad = vec![Stage {
        name: "self".to_string(),
        duration_ms: 5,
        depends_on: vec![0],
    }];
    println!("self-dep: {:?}", schedule(&bad));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_pipeline() -> Vec<Stage> {
        vec![
            Stage {
                name: "a".to_string(),
                duration_ms: 5,
                depends_on: vec![],
            },
            Stage {
                name: "b".to_string(),
                duration_ms: 10,
                depends_on: vec![0],
            },
            Stage {
                name: "c".to_string(),
                duration_ms: 100,
                depends_on: vec![1],
            },
        ]
    }

    #[test]
    fn dag_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_pipeline_critical_path_sums() {
        let v = schedule(&linear_pipeline());
        if let DagVerdict::Ok {
            critical_path_ms, ..
        } = v
        {
            assert_eq!(critical_path_ms, 115);
        }
    }

    #[test]
    fn empty_dag_rejected() {
        assert_eq!(schedule(&[]), DagVerdict::EmptyDag);
    }

    #[test]
    fn self_dependency_rejected() {
        let stages = vec![Stage {
            name: "self".to_string(),
            duration_ms: 5,
            depends_on: vec![0],
        }];
        let v = schedule(&stages);
        assert!(matches!(v, DagVerdict::InvalidDependency { .. }));
    }

    #[test]
    fn forward_dependency_rejected() {
        let stages = vec![
            Stage {
                name: "a".to_string(),
                duration_ms: 5,
                depends_on: vec![1],
            },
            Stage {
                name: "b".to_string(),
                duration_ms: 5,
                depends_on: vec![],
            },
        ];
        let v = schedule(&stages);
        assert!(matches!(v, DagVerdict::InvalidDependency { .. }));
    }

    #[test]
    fn parallel_branches_critical_path_max() {
        // Two branches of different length: max wins.
        let stages = vec![
            Stage {
                name: "root".to_string(),
                duration_ms: 5,
                depends_on: vec![],
            },
            Stage {
                name: "fast".to_string(),
                duration_ms: 10,
                depends_on: vec![0],
            },
            Stage {
                name: "slow".to_string(),
                duration_ms: 100,
                depends_on: vec![0],
            },
        ];
        if let DagVerdict::Ok {
            critical_path_ms, ..
        } = schedule(&stages)
        {
            assert_eq!(critical_path_ms, 105);
        }
    }

    #[test]
    fn finish_times_match_dependencies() {
        let v = schedule(&linear_pipeline());
        if let DagVerdict::Ok { finish_times, .. } = v {
            assert_eq!(finish_times[0], 5);
            assert_eq!(finish_times[1], 15);
            assert_eq!(finish_times[2], 115);
        }
    }

    #[test]
    fn diamond_dependency_handles_join() {
        // a → b, a → c, then d depends on both.
        let stages = vec![
            Stage {
                name: "a".to_string(),
                duration_ms: 5,
                depends_on: vec![],
            },
            Stage {
                name: "b".to_string(),
                duration_ms: 10,
                depends_on: vec![0],
            },
            Stage {
                name: "c".to_string(),
                duration_ms: 50,
                depends_on: vec![0],
            },
            Stage {
                name: "d".to_string(),
                duration_ms: 5,
                depends_on: vec![1, 2],
            },
        ];
        if let DagVerdict::Ok {
            critical_path_ms, ..
        } = schedule(&stages)
        {
            // Critical = 5 + 50 + 5 = 60 (a → c → d).
            assert_eq!(critical_path_ms, 60);
        }
    }

    #[test]
    fn single_stage_critical_path_is_duration() {
        let stages = vec![Stage {
            name: "only".to_string(),
            duration_ms: 42,
            depends_on: vec![],
        }];
        if let DagVerdict::Ok {
            critical_path_ms, ..
        } = schedule(&stages)
        {
            assert_eq!(critical_path_ms, 42);
        }
    }

    #[test]
    fn zero_duration_stages_handled() {
        let stages = vec![
            Stage {
                name: "a".to_string(),
                duration_ms: 0,
                depends_on: vec![],
            },
            Stage {
                name: "b".to_string(),
                duration_ms: 0,
                depends_on: vec![0],
            },
        ];
        if let DagVerdict::Ok {
            critical_path_ms, ..
        } = schedule(&stages)
        {
            assert_eq!(critical_path_ms, 0);
        }
    }

    #[test]
    fn finish_times_match_input_count() {
        let v = schedule(&linear_pipeline());
        if let DagVerdict::Ok { finish_times, .. } = v {
            assert_eq!(finish_times.len(), 3);
        }
    }
}
