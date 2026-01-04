# Distributed Inference

Multi-node inference with repartir work-stealing scheduler.

## Example

```bash
cargo run --example distributed_inference
```

## Code

```rust
//! Distributed Inference Example
//!
//! Demonstrates multi-node inference using repartir.

use apr_cookbook::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Distributed Inference Example ===\n");

    // Configuration
    let config = InferenceConfig {
        num_shards: 4,
        num_workers: 4,
        batch_size: 32,
        embed_dim: 768,
    };

    println!("1. Configuration");
    println!("   Shards:        {}", config.num_shards);
    println!("   Workers:       {}", config.num_workers);
    println!("   Batch size:    {}", config.batch_size);
    println!("   Embed dim:     {}", config.embed_dim);
    println!();

    // Create distributed inference engine
    let engine = DistributedInference::new(config.clone());

    println!("2. Model Sharding");
    println!("   Created {} shards", engine.shards.len());
    println!("   FLOPS per sample: {}", engine.total_flops());
    println!();

    // Run inference
    println!("3. Inference Demo");
    let test_input: Vec<f32> = (0..config.embed_dim)
        .map(|i| (i as f32).sin())
        .collect();

    let results = engine.infer(std::slice::from_ref(&test_input));

    if let Some(output) = results.first() {
        println!("   Input:  [{:.4}, {:.4}, ...]", test_input[0], test_input[1]);
        println!("   Output: [{:.4}, {:.4}, ...]", output[0], output[1]);
    }
    println!();

    // Benchmark
    println!("4. Benchmark");
    println!("   ┌──────────────┬────────────┬────────────┐");
    println!("   │ Shards       │ Samples/s  │ GFLOPS     │");
    println!("   ├──────────────┼────────────┼────────────┤");

    for num_shards in [1, 2, 4, 8] {
        let bench_config = InferenceConfig { num_shards, ..config.clone() };
        let result = run_benchmark(&bench_config, 10);
        println!("   │ {:12} │ {:10.1} │ {:10.4} │",
            num_shards, result.samples_per_sec, result.gflops);
    }
    println!("   └──────────────┴────────────┴────────────┘");

    println!("\n=== Example Complete ===");
    Ok(())
}
```

## Key Features

### Model Sharding

Distribute model across multiple workers:

```rust
// Shard model across 4 workers
let shards: Vec<_> = (0..4)
    .map(|i| ModelShard::new(i, 4, embed_dim))
    .collect();

// Pipeline parallel execution
for shard in &shards {
    intermediate = shard.forward(&intermediate);
}
```

### Work-Stealing Scheduler

Automatically balance load across workers:

```rust
use repartir::{Pool, Scheduler};

let pool = Pool::builder()
    .scheduler(Scheduler::WorkStealing)
    .cpu_workers(8)
    .build()?;

// Idle workers steal from busy ones
pool.submit_batch(tasks).await?;
```

### Remote Execution

Distribute across multiple machines:

```rust
use repartir::executor::remote::RemoteExecutor;

let executor = RemoteExecutor::builder()
    .add_worker("node1:9000")
    .add_worker("node2:9000")
    .add_worker("node3:9000")
    .build().await?;

let results = executor.execute_batch(tasks).await?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     repartir Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────────────────────────────────┐ │
│  │  Tasks  │───►│            Scheduler                    │ │
│  └─────────┘    │   ┌───────────────────────────────────┐ │ │
│                 │   │  Deque[0]  Deque[1]  ...  Deque[N] │ │ │
│                 │   └───────────────────────────────────┘ │ │
│                 │              ▼ steal ▲                   │ │
│                 └─────────────────────────────────────────┘ │
│                              │                              │
│                 ┌────────────┼────────────┐                │
│                 ▼            ▼            ▼                │
│           ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│           │Worker 0 │  │Worker 1 │  │Worker N │           │
│           │  (CPU)  │  │  (GPU)  │  │(Remote) │           │
│           └─────────┘  └─────────┘  └─────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Performance

| Configuration | Throughput | Efficiency |
|---------------|------------|------------|
| 1 shard, 1 worker | 100 samples/s | 100% |
| 4 shards, 4 workers | 380 samples/s | 95% |
| 8 shards, 8 workers | 720 samples/s | 90% |

## Tests

```rust
#[test]
fn test_distributed_inference_creation() {
    let config = InferenceConfig::default();
    let engine = DistributedInference::new(config.clone());
    assert_eq!(engine.shards.len(), config.num_shards);
}

#[test]
fn test_distributed_inference_infer() {
    let config = InferenceConfig {
        num_shards: 2,
        embed_dim: 64,
        ..Default::default()
    };
    let engine = DistributedInference::new(config.clone());
    let inputs = vec![vec![1.0f32; 64]; 4];
    let outputs = engine.infer(&inputs);

    assert_eq!(outputs.len(), 4);
    for output in &outputs {
        assert_eq!(output.len(), 64);
    }
}

#[test]
fn test_work_stealing_load_balance() {
    // Work-stealing should balance uneven loads
    let pool = Pool::builder()
        .cpu_workers(4)
        .build()
        .unwrap();

    let mut tasks = Vec::new();
    // Create imbalanced workload
    for i in 0..100 {
        let duration = if i % 10 == 0 { 100 } else { 10 }; // Heavy task every 10th
        tasks.push(Task::new(move || std::thread::sleep(Duration::from_millis(duration))));
    }

    let start = Instant::now();
    pool.execute_all(tasks);
    let elapsed = start.elapsed();

    // Should complete faster than sequential due to stealing
    assert!(elapsed.as_millis() < 500);
}
```
