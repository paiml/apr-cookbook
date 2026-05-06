//! # Advanced Inference-Replica Failover Picker
//!
//! When a replica goes down, pick its successor from healthy replicas
//! by: lowest p99 latency among those at < 75% utilization. Tie-break
//! by lower active-request count.
//!
//! If all healthy replicas are over-utilized, return AllOverloaded.
//!
//! Demonstrates the **ADV.14** recipe for PMAT-142 (advanced round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Envoy load-balancer least-request + outlier-detection rules.
//!
//! Run with: cargo run --example adv_replica_failover
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Healthy,
    Degraded,
    Down,
}

#[derive(Debug, Clone)]
pub struct Replica {
    pub name: String,
    pub health: Health,
    pub utilization_pct: u32,
    pub p99_ms: u32,
    pub active_requests: u32,
}

#[derive(Debug, PartialEq)]
pub enum FailoverVerdict {
    Ok { successor: String },
    AllOverloaded,
    NoHealthyReplicas,
    EmptyPool,
}

const UTILIZATION_CAP: u32 = 75;

pub fn pick(replicas: &[Replica]) -> FailoverVerdict {
    if replicas.is_empty() {
        return FailoverVerdict::EmptyPool;
    }
    let healthy: Vec<&Replica> = replicas
        .iter()
        .filter(|r| r.health == Health::Healthy)
        .collect();
    if healthy.is_empty() {
        return FailoverVerdict::NoHealthyReplicas;
    }
    let under_capacity: Vec<&Replica> = healthy
        .iter()
        .copied()
        .filter(|r| r.utilization_pct < UTILIZATION_CAP)
        .collect();
    if under_capacity.is_empty() {
        return FailoverVerdict::AllOverloaded;
    }
    let best = under_capacity
        .iter()
        .copied()
        .min_by(|a, b| {
            a.p99_ms
                .cmp(&b.p99_ms)
                .then(a.active_requests.cmp(&b.active_requests))
        })
        .unwrap();
    FailoverVerdict::Ok {
        successor: best.name.clone(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_replica_failover")?;

    let pool = vec![
        Replica {
            name: "r1".to_string(),
            health: Health::Healthy,
            utilization_pct: 30,
            p99_ms: 100,
            active_requests: 10,
        },
        Replica {
            name: "r2".to_string(),
            health: Health::Healthy,
            utilization_pct: 50,
            p99_ms: 80,
            active_requests: 8,
        },
        Replica {
            name: "r3".to_string(),
            health: Health::Down,
            utilization_pct: 0,
            p99_ms: 0,
            active_requests: 0,
        },
    ];
    println!("typical: {:?}", pick(&pool));

    let all_overloaded: Vec<Replica> = pool
        .iter()
        .map(|r| Replica {
            utilization_pct: 90,
            ..r.clone()
        })
        .collect();
    println!("all overloaded: {:?}", pick(&all_overloaded));

    println!("empty: {:?}", pick(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replica(name: &str, util: u32, p99: u32, active: u32) -> Replica {
        Replica {
            name: name.to_string(),
            health: Health::Healthy,
            utilization_pct: util,
            p99_ms: p99,
            active_requests: active,
        }
    }

    #[test]
    fn failover_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn picks_lowest_p99() {
        let pool = vec![
            replica("a", 30, 100, 5),
            replica("b", 30, 50, 5),
            replica("c", 30, 200, 5),
        ];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "b");
        }
    }

    #[test]
    fn tiebreak_by_lower_active() {
        let pool = vec![replica("a", 30, 100, 10), replica("b", 30, 100, 5)];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "b");
        }
    }

    #[test]
    fn skips_overloaded() {
        let pool = vec![replica("a", 90, 50, 5), replica("b", 30, 100, 5)];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "b");
        }
    }

    #[test]
    fn all_overloaded_reported() {
        let pool = vec![replica("a", 90, 50, 5), replica("b", 95, 60, 5)];
        assert_eq!(pick(&pool), FailoverVerdict::AllOverloaded);
    }

    #[test]
    fn no_healthy_reported() {
        let pool = vec![Replica {
            name: "down".to_string(),
            health: Health::Down,
            utilization_pct: 0,
            p99_ms: 0,
            active_requests: 0,
        }];
        assert_eq!(pick(&pool), FailoverVerdict::NoHealthyReplicas);
    }

    #[test]
    fn empty_pool_rejected() {
        assert_eq!(pick(&[]), FailoverVerdict::EmptyPool);
    }

    #[test]
    fn degraded_treated_as_unhealthy() {
        let pool = vec![Replica {
            name: "r".to_string(),
            health: Health::Degraded,
            utilization_pct: 30,
            p99_ms: 50,
            active_requests: 5,
        }];
        assert_eq!(pick(&pool), FailoverVerdict::NoHealthyReplicas);
    }

    #[test]
    fn at_75_utilization_excluded() {
        // Strict less-than rule: 75% counts as overloaded.
        let pool = vec![replica("a", 75, 50, 5), replica("b", 30, 100, 5)];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "b");
        }
    }

    #[test]
    fn just_under_75_admitted() {
        let pool = vec![replica("a", 74, 50, 5)];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "a");
        }
    }

    #[test]
    fn single_healthy_picked() {
        let pool = vec![replica("only", 30, 100, 5)];
        if let FailoverVerdict::Ok { successor } = pick(&pool) {
            assert_eq!(successor, "only");
        }
    }
}
