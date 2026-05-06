//! # Deployment Pod Replica Scheduler
//!
//! Schedule N pod replicas across M nodes balancing CPU + memory:
//! pods_per_node = ceil(N / M); ensure each node has CPU/RAM headroom
//! for assigned pods. This recipe builds the scheduler + per-node
//! resource fit check.
//!
//! Demonstrates the **DEPLOY.15** recipe for PMAT-130 (deployment-stacks coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kubernetes scheduling docs § Pod Spread Constraints.
//!
//! Run with: cargo run --example deploy_pod_replica_scheduler
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct PodRequest {
    pub cpu_millis: u32,
    pub mem_mib: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct NodeCapacity {
    pub cpu_millis: u32,
    pub mem_mib: u32,
}

#[derive(Debug, PartialEq)]
pub enum SchedulerVerdict {
    Ok { assignments: Vec<u32> },
    NoNodes,
    NoReplicas,
    PodExceedsAnyNode { pod_index: usize },
    InsufficientCapacity { pod_index: usize, node_index: usize },
}

pub fn schedule(replicas: u32, pod: PodRequest, nodes: &[NodeCapacity]) -> SchedulerVerdict {
    if replicas == 0 {
        return SchedulerVerdict::NoReplicas;
    }
    if nodes.is_empty() {
        return SchedulerVerdict::NoNodes;
    }
    let mut remaining_cpu: Vec<i64> = nodes.iter().map(|n| i64::from(n.cpu_millis)).collect();
    let mut remaining_mem: Vec<i64> = nodes.iter().map(|n| i64::from(n.mem_mib)).collect();
    let mut counts = vec![0u32; nodes.len()];
    for i in 0..replicas {
        // Bin-pack: pick node with most remaining CPU.
        let (best_idx, _) = remaining_cpu
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .unwrap();
        let needed_cpu = i64::from(pod.cpu_millis);
        let needed_mem = i64::from(pod.mem_mib);
        if remaining_cpu[best_idx] < needed_cpu || remaining_mem[best_idx] < needed_mem {
            // Check if any node ever could fit this pod.
            let any_fits = nodes
                .iter()
                .any(|n| n.cpu_millis >= pod.cpu_millis && n.mem_mib >= pod.mem_mib);
            if !any_fits {
                return SchedulerVerdict::PodExceedsAnyNode {
                    pod_index: i as usize,
                };
            }
            return SchedulerVerdict::InsufficientCapacity {
                pod_index: i as usize,
                node_index: best_idx,
            };
        }
        remaining_cpu[best_idx] -= needed_cpu;
        remaining_mem[best_idx] -= needed_mem;
        counts[best_idx] += 1;
    }
    SchedulerVerdict::Ok {
        assignments: counts,
    }
}

pub fn balance_delta(assignments: &[u32]) -> u32 {
    let max = assignments.iter().copied().max().unwrap_or(0);
    let min = assignments.iter().copied().min().unwrap_or(0);
    max - min
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("deploy_pod_replica_scheduler")?;

    let nodes = vec![
        NodeCapacity {
            cpu_millis: 4000,
            mem_mib: 8192,
        },
        NodeCapacity {
            cpu_millis: 4000,
            mem_mib: 8192,
        },
        NodeCapacity {
            cpu_millis: 4000,
            mem_mib: 8192,
        },
    ];
    let pod = PodRequest {
        cpu_millis: 500,
        mem_mib: 1024,
    };
    println!("3 nodes × 8 pods: {:?}", schedule(8, pod, &nodes));
    println!("0 replicas: {:?}", schedule(0, pod, &nodes));
    println!("0 nodes: {:?}", schedule(1, pod, &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_nodes() -> Vec<NodeCapacity> {
        vec![
            NodeCapacity {
                cpu_millis: 4000,
                mem_mib: 8192,
            },
            NodeCapacity {
                cpu_millis: 4000,
                mem_mib: 8192,
            },
        ]
    }

    fn small_pod() -> PodRequest {
        PodRequest {
            cpu_millis: 500,
            mem_mib: 1024,
        }
    }

    #[test]
    fn scheduler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn even_pods_balanced() {
        if let SchedulerVerdict::Ok { assignments } = schedule(4, small_pod(), &standard_nodes()) {
            // 4 pods × 2 nodes → 2 each, balanced.
            assert_eq!(assignments.len(), 2);
            assert_eq!(balance_delta(&assignments), 0);
        }
    }

    #[test]
    fn odd_pods_at_most_one_off() {
        if let SchedulerVerdict::Ok { assignments } = schedule(5, small_pod(), &standard_nodes()) {
            assert!(balance_delta(&assignments) <= 1);
        }
    }

    #[test]
    fn no_replicas_rejected() {
        assert_eq!(
            schedule(0, small_pod(), &standard_nodes()),
            SchedulerVerdict::NoReplicas
        );
    }

    #[test]
    fn no_nodes_rejected() {
        assert_eq!(schedule(1, small_pod(), &[]), SchedulerVerdict::NoNodes);
    }

    #[test]
    fn pod_too_big_rejected() {
        let huge = PodRequest {
            cpu_millis: 100_000,
            mem_mib: 100_000,
        };
        let v = schedule(1, huge, &standard_nodes());
        assert!(matches!(
            v,
            SchedulerVerdict::PodExceedsAnyNode { pod_index: 0 }
        ));
    }

    #[test]
    fn capacity_exhaustion_detected() {
        // Each node can fit 8 pods of 500m × 1024 MiB. Try 100 pods.
        let v = schedule(100, small_pod(), &standard_nodes());
        assert!(matches!(v, SchedulerVerdict::InsufficientCapacity { .. }));
    }

    #[test]
    fn each_pod_fits_into_max_capacity() {
        // 16 pods × 2 nodes (each fits 8) → all placed.
        let v = schedule(16, small_pod(), &standard_nodes());
        assert!(matches!(v, SchedulerVerdict::Ok { .. }));
    }

    #[test]
    fn balance_delta_helper_works() {
        assert_eq!(balance_delta(&[3, 3, 3]), 0);
        assert_eq!(balance_delta(&[3, 4, 5]), 2);
        assert_eq!(balance_delta(&[]), 0);
    }

    #[test]
    fn single_node_takes_everything() {
        let nodes = vec![NodeCapacity {
            cpu_millis: 8000,
            mem_mib: 16384,
        }];
        if let SchedulerVerdict::Ok { assignments } = schedule(4, small_pod(), &nodes) {
            assert_eq!(assignments, vec![4]);
        }
    }
}
