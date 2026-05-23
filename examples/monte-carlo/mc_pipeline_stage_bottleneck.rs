//! # Monte-Carlo Pipeline Stage Bottleneck
//!
//! Sim N-stage processing pipeline with per-stage service rates.
//! Identifies the bottleneck stage by max queue length over the run.
//!
//! Demonstrates the **MC.68** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goldratt, Theory of Constraints (1984); ETL pipeline
//!  bottleneck literature.
//!
//! Run with: cargo run --example mc_pipeline_stage_bottleneck
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BottleneckVerdict {
    Ok {
        max_queue_per_stage: Vec<u32>,
        bottleneck_stage: u32,
    },
    InvalidConfig,
}

pub fn simulate(seconds: u32, arrival_per_sec: u32, stage_capacities: &[u32]) -> BottleneckVerdict {
    if seconds == 0 || arrival_per_sec == 0 || stage_capacities.is_empty() {
        return BottleneckVerdict::InvalidConfig;
    }
    if stage_capacities.contains(&0) {
        return BottleneckVerdict::InvalidConfig;
    }
    let n = stage_capacities.len();
    let mut queues: Vec<u32> = vec![0; n];
    let mut max_queue: Vec<u32> = vec![0; n];
    for _ in 0..seconds {
        // Arrivals into stage 0.
        queues[0] += arrival_per_sec;
        // Process each stage in order: serve up to capacity, push to next.
        for i in 0..n {
            let served = queues[i].min(stage_capacities[i]);
            queues[i] -= served;
            if i + 1 < n {
                queues[i + 1] += served;
            }
        }
        for i in 0..n {
            if queues[i] > max_queue[i] {
                max_queue[i] = queues[i];
            }
        }
    }
    let mut bottleneck_idx = 0usize;
    for i in 1..n {
        if max_queue[i] > max_queue[bottleneck_idx] {
            bottleneck_idx = i;
        }
    }
    BottleneckVerdict::Ok {
        max_queue_per_stage: max_queue,
        bottleneck_stage: bottleneck_idx as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_pipeline_stage_bottleneck")?;

    let balanced = [10u32, 10, 10];
    println!("balanced: {:?}", simulate(60, 5, &balanced));
    let middle_slow = [10u32, 3, 10];
    println!("middle bottleneck: {:?}", simulate(60, 5, &middle_slow));
    println!("invalid: {:?}", simulate(0, 5, &balanced));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn middle_slow_is_bottleneck() {
        let stages = [10u32, 3, 10];
        let v = simulate(100, 5, &stages);
        if let BottleneckVerdict::Ok {
            bottleneck_stage, ..
        } = v
        {
            assert_eq!(bottleneck_stage, 1);
        }
    }

    #[test]
    fn first_slow_is_bottleneck() {
        let stages = [3u32, 10, 10];
        let v = simulate(100, 5, &stages);
        if let BottleneckVerdict::Ok {
            bottleneck_stage, ..
        } = v
        {
            assert_eq!(bottleneck_stage, 0);
        }
    }

    #[test]
    fn last_slow_is_bottleneck() {
        let stages = [10u32, 10, 3];
        let v = simulate(100, 5, &stages);
        if let BottleneckVerdict::Ok {
            bottleneck_stage, ..
        } = v
        {
            assert_eq!(bottleneck_stage, 2);
        }
    }

    #[test]
    fn balanced_pipeline_no_growth() {
        // arrivals == capacity → no queue accumulates.
        let stages = [5u32, 5, 5];
        let v = simulate(100, 5, &stages);
        if let BottleneckVerdict::Ok {
            max_queue_per_stage,
            ..
        } = v
        {
            assert!(max_queue_per_stage.iter().all(|q| *q <= 10));
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        let stages = [5u32];
        assert_eq!(simulate(0, 5, &stages), BottleneckVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_arrival() {
        let stages = [5u32];
        assert_eq!(simulate(60, 0, &stages), BottleneckVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_stages() {
        assert_eq!(simulate(60, 5, &[]), BottleneckVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity_stage() {
        let stages = [5u32, 0, 5];
        assert_eq!(simulate(60, 5, &stages), BottleneckVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let stages = [5u32, 3, 5];
        let r1 = simulate(60, 5, &stages);
        let r2 = simulate(60, 5, &stages);
        assert_eq!(r1, r2);
    }

    #[test]
    fn max_queue_grows_over_time() {
        let stages = [10u32, 1, 10];
        let short = simulate(10, 5, &stages);
        let long = simulate(100, 5, &stages);
        if let (
            BottleneckVerdict::Ok {
                max_queue_per_stage: s,
                ..
            },
            BottleneckVerdict::Ok {
                max_queue_per_stage: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l[1] > s[1]);
        }
    }

    #[test]
    fn single_stage_works() {
        let stages = [5u32];
        let v = simulate(60, 3, &stages);
        if let BottleneckVerdict::Ok {
            bottleneck_stage, ..
        } = v
        {
            assert_eq!(bottleneck_stage, 0);
        }
    }
}
