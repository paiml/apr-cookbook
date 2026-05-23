//! # Monitoring Correlated Failures Detector
//!
//! When two services fail within a small time window, they're likely
//! correlated (cascading failure or shared dependency). Detector:
//! given timestamped failures, group by (window_secs) → list
//! correlated clusters.
//!
//! Demonstrates the **MON.34** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Causal-impact analysis for service outages.
//!
//! Run with: cargo run --example monitor_correlated_failures
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureEvent {
    pub service: String,
    pub timestamp_secs: u64,
}

#[derive(Debug, PartialEq)]
pub enum CorrelationVerdict {
    Ok {
        clusters: Vec<Vec<String>>,
        max_cluster_size: usize,
    },
    EmptyEvents,
    InvalidWindow,
}

pub fn detect(events: &[FailureEvent], window_secs: u64) -> CorrelationVerdict {
    if events.is_empty() {
        return CorrelationVerdict::EmptyEvents;
    }
    if window_secs == 0 {
        return CorrelationVerdict::InvalidWindow;
    }
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|e| e.timestamp_secs);
    let mut clusters: Vec<Vec<FailureEvent>> = Vec::new();
    for e in sorted {
        match clusters.last_mut() {
            Some(c)
                if !c.is_empty()
                    && e.timestamp_secs
                        .saturating_sub(c.last().unwrap().timestamp_secs)
                        <= window_secs =>
            {
                c.push(e);
            }
            _ => clusters.push(vec![e]),
        }
    }
    let services_only: Vec<Vec<String>> = clusters
        .iter()
        .filter(|c| c.len() > 1)
        .map(|c| c.iter().map(|e| e.service.clone()).collect())
        .collect();
    let max_cluster_size = services_only.iter().map(Vec::len).max().unwrap_or(0);
    CorrelationVerdict::Ok {
        clusters: services_only,
        max_cluster_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_correlated_failures")?;

    let events = vec![
        FailureEvent {
            service: "auth".to_string(),
            timestamp_secs: 100,
        },
        FailureEvent {
            service: "api".to_string(),
            timestamp_secs: 105,
        },
        FailureEvent {
            service: "db".to_string(),
            timestamp_secs: 200,
        },
    ];
    println!("typical: {:?}", detect(&events, 30));
    println!("empty: {:?}", detect(&[], 30));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(service: &str, ts: u64) -> FailureEvent {
        FailureEvent {
            service: service.to_string(),
            timestamp_secs: ts,
        }
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn close_failures_clustered() {
        let events = vec![ev("auth", 100), ev("api", 105)];
        let v = detect(&events, 30);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            assert_eq!(clusters.len(), 1);
            assert_eq!(clusters[0].len(), 2);
        }
    }

    #[test]
    fn distant_failures_separate() {
        let events = vec![ev("a", 100), ev("b", 1000)];
        let v = detect(&events, 30);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            // No clusters >1.
            assert!(clusters.is_empty());
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(detect(&[], 30), CorrelationVerdict::EmptyEvents);
    }

    #[test]
    fn zero_window_rejected() {
        let events = vec![ev("a", 100)];
        assert_eq!(detect(&events, 0), CorrelationVerdict::InvalidWindow);
    }

    #[test]
    fn three_in_chain_clustered() {
        let events = vec![ev("a", 100), ev("b", 110), ev("c", 120)];
        let v = detect(&events, 15);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            assert_eq!(clusters[0].len(), 3);
        }
    }

    #[test]
    fn max_cluster_size_returned() {
        let events = vec![ev("a", 100), ev("b", 110), ev("c", 120)];
        let v = detect(&events, 15);
        if let CorrelationVerdict::Ok {
            max_cluster_size, ..
        } = v
        {
            assert_eq!(max_cluster_size, 3);
        }
    }

    #[test]
    fn out_of_order_input_sorted() {
        let events = vec![ev("c", 120), ev("a", 100), ev("b", 110)];
        let v = detect(&events, 15);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            // First cluster should be a, b, c in time order.
            assert_eq!(clusters[0], vec!["a", "b", "c"]);
        }
    }

    #[test]
    fn boundary_at_window_clustered() {
        // Exactly at window boundary (30s gap, window=30) → clustered.
        let events = vec![ev("a", 100), ev("b", 130)];
        let v = detect(&events, 30);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            assert_eq!(clusters.len(), 1);
        }
    }

    #[test]
    fn just_outside_window_not_clustered() {
        let events = vec![ev("a", 100), ev("b", 131)];
        let v = detect(&events, 30);
        if let CorrelationVerdict::Ok {
            max_cluster_size, ..
        } = v
        {
            assert_eq!(max_cluster_size, 0);
        }
    }

    #[test]
    fn singleton_failures_not_in_clusters() {
        let events = vec![ev("a", 100)];
        let v = detect(&events, 30);
        if let CorrelationVerdict::Ok { clusters, .. } = v {
            assert!(clusters.is_empty());
        }
    }
}
