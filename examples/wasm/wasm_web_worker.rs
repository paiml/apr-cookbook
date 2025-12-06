//! # Recipe: Web Worker Inference
//!
//! **Category**: WASM/Browser
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (Verified)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Offload inference to Web Worker for non-blocking UI.
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_web_worker
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_web_worker")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Web Worker inference simulation");
    println!();

    // Create worker pool
    let mut pool = WorkerPool::new(4);
    ctx.record_metric("worker_count", pool.workers.len() as i64);

    println!("Worker Pool:");
    println!("  Workers: {}", pool.workers.len());
    println!();

    // Queue inference tasks
    let tasks = vec![
        InferenceTask {
            id: 1,
            inputs: vec![0.5, 0.3, 0.8, 0.2],
        },
        InferenceTask {
            id: 2,
            inputs: vec![0.1, 0.9, 0.2, 0.4],
        },
        InferenceTask {
            id: 3,
            inputs: vec![0.7, 0.2, 0.5, 0.6],
        },
        InferenceTask {
            id: 4,
            inputs: vec![0.3, 0.4, 0.1, 0.8],
        },
        InferenceTask {
            id: 5,
            inputs: vec![0.9, 0.1, 0.3, 0.5],
        },
        InferenceTask {
            id: 6,
            inputs: vec![0.2, 0.6, 0.9, 0.1],
        },
    ];

    println!("Queuing {} tasks...", tasks.len());
    for task in &tasks {
        pool.queue_task(task.clone());
    }
    ctx.record_metric("tasks_queued", tasks.len() as i64);

    // Process tasks
    println!();
    println!("Processing tasks:");
    println!("{:-<60}", "");
    println!(
        "{:<8} {:<10} {:>12} {:>15}",
        "Task", "Worker", "Duration", "Status"
    );
    println!("{:-<60}", "");

    let results = pool.process_all();

    for result in &results {
        println!(
            "{:<8} {:<10} {:>10}ms {:>15}",
            format!("#{}", result.task_id),
            format!("W{}", result.worker_id),
            result.duration_ms,
            if result.success {
                "completed"
            } else {
                "failed"
            }
        );
    }
    println!("{:-<60}", "");

    // Statistics
    let total_duration: u32 = results.iter().map(|r| r.duration_ms).sum();
    let parallel_time = results.iter().map(|r| r.duration_ms).max().unwrap_or(0);

    ctx.record_metric("total_duration_ms", i64::from(total_duration));
    ctx.record_metric("parallel_time_ms", i64::from(parallel_time));

    let speedup = f64::from(total_duration) / f64::from(parallel_time);

    println!();
    println!("Performance:");
    println!("  Sequential time: {}ms", total_duration);
    println!("  Parallel time: {}ms", parallel_time);
    println!("  Speedup: {:.2}x", speedup);
    println!(
        "  Efficiency: {:.1}%",
        (speedup / pool.workers.len() as f64) * 100.0
    );

    // Save results
    let results_path = ctx.path("worker_results.json");
    save_results(&results_path, &results)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceTask {
    id: u32,
    inputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskResult {
    task_id: u32,
    worker_id: u32,
    outputs: Vec<f32>,
    duration_ms: u32,
    success: bool,
}

#[derive(Debug)]
#[allow(dead_code)]
struct Worker {
    id: u32,
    busy: bool,
}

#[derive(Debug)]
struct WorkerPool {
    workers: Vec<Worker>,
    task_queue: VecDeque<InferenceTask>,
}

impl WorkerPool {
    fn new(num_workers: u32) -> Self {
        let workers = (0..num_workers)
            .map(|id| Worker { id, busy: false })
            .collect();

        Self {
            workers,
            task_queue: VecDeque::new(),
        }
    }

    fn queue_task(&mut self, task: InferenceTask) {
        self.task_queue.push_back(task);
    }

    fn process_all(&mut self) -> Vec<TaskResult> {
        let mut results = Vec::new();
        let mut worker_idx = 0;
        let num_workers = self.workers.len();

        while let Some(task) = self.task_queue.pop_front() {
            let worker = &mut self.workers[worker_idx % num_workers];
            let result = Self::execute_task(worker, &task);
            results.push(result);
            worker_idx += 1;
        }

        results
    }

    fn execute_task(worker: &Worker, task: &InferenceTask) -> TaskResult {
        // Deterministic mock inference
        let outputs: Vec<f32> = task.inputs.iter().map(|x| (x * 2.0).tanh()).collect();

        // Deterministic duration based on task id and worker id
        let duration = 10 + (task.id * 3 + worker.id) % 20;

        TaskResult {
            task_id: task.id,
            worker_id: worker.id,
            outputs,
            duration_ms: duration,
            success: true,
        }
    }
}

fn save_results(path: &std::path::Path, results: &[TaskResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_pool_creation() {
        let pool = WorkerPool::new(4);
        assert_eq!(pool.workers.len(), 4);
        assert!(pool.task_queue.is_empty());
    }

    #[test]
    fn test_queue_task() {
        let mut pool = WorkerPool::new(2);
        pool.queue_task(InferenceTask {
            id: 1,
            inputs: vec![0.5],
        });

        assert_eq!(pool.task_queue.len(), 1);
    }

    #[test]
    fn test_process_all() {
        let mut pool = WorkerPool::new(2);
        pool.queue_task(InferenceTask {
            id: 1,
            inputs: vec![0.5],
        });
        pool.queue_task(InferenceTask {
            id: 2,
            inputs: vec![0.3],
        });

        let results = pool.process_all();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_worker_assignment() {
        let mut pool = WorkerPool::new(2);
        pool.queue_task(InferenceTask {
            id: 1,
            inputs: vec![0.5],
        });
        pool.queue_task(InferenceTask {
            id: 2,
            inputs: vec![0.3],
        });
        pool.queue_task(InferenceTask {
            id: 3,
            inputs: vec![0.7],
        });

        let results = pool.process_all();

        // Tasks should be distributed round-robin
        assert_eq!(results[0].worker_id, 0);
        assert_eq!(results[1].worker_id, 1);
        assert_eq!(results[2].worker_id, 0);
    }

    #[test]
    fn test_deterministic_duration() {
        let worker = Worker { id: 0, busy: false };
        let task = InferenceTask {
            id: 1,
            inputs: vec![0.5],
        };

        let r1 = WorkerPool::execute_task(&worker, &task);
        let r2 = WorkerPool::execute_task(&worker, &task);

        assert_eq!(r1.duration_ms, r2.duration_ms);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_worker_results").unwrap();
        let path = ctx.path("results.json");

        let results = vec![TaskResult {
            task_id: 1,
            worker_id: 0,
            outputs: vec![0.5],
            duration_ms: 10,
            success: true,
        }];

        save_results(&path, &results).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_all_tasks_processed(n_tasks in 1usize..20, n_workers in 1u32..8) {
            let mut pool = WorkerPool::new(n_workers);

            for i in 0..n_tasks {
                pool.queue_task(InferenceTask {
                    id: i as u32,
                    inputs: vec![0.5],
                });
            }

            let results = pool.process_all();
            prop_assert_eq!(results.len(), n_tasks);
        }

        #[test]
        fn prop_all_succeed(n_tasks in 1usize..10) {
            let mut pool = WorkerPool::new(4);

            for i in 0..n_tasks {
                pool.queue_task(InferenceTask {
                    id: i as u32,
                    inputs: vec![0.5],
                });
            }

            let results = pool.process_all();
            prop_assert!(results.iter().all(|r| r.success));
        }
    }
}
