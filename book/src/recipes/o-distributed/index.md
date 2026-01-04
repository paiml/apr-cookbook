# Category O: Distributed Computing

This category covers distributed inference using repartir, a work-stealing scheduler for multi-node ML workloads.

## Recipes

| Recipe | Description |
|--------|-------------|
| [Distributed Inference](./distributed-inference.md) | Multi-node inference with repartir |

## Key Concepts

### repartir

A distributed computing library featuring:
- **Work-Stealing Scheduler**: Blumofe & Leiserson (1999) algorithm
- **CPU Executor**: Local multi-core parallel execution
- **GPU Executor**: wgpu-based GPU compute
- **Remote Executor**: TCP-based distributed execution

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Distributed Inference Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │  Tasks  │ ─► │  Scheduler  │ ─► │  Workers (CPU/GPU)      │  │
│  │ (batch) │    │ (steal-work)│    │  ├── worker-0          │  │
│  └─────────┘    └─────────────┘    │  ├── worker-1          │  │
│                                     │  ├── worker-2          │  │
│                                     │  └── worker-N          │  │
│                                     └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Stack Integration

```rust
use repartir::{Pool, task::{Task, Backend}};

// Create worker pool
let pool = Pool::builder()
    .cpu_workers(8)
    .max_queue_size(1000)
    .build()?;

// Submit tasks for parallel execution
let task = Task::builder()
    .name("inference")
    .data(input_data)
    .backend(Backend::Cpu)
    .build()?;

let result = pool.submit(task).await?;
```

## Feature Flags

| Feature | Purpose |
|---------|---------|
| `cpu` (default) | Local multi-core execution |
| `gpu` | wgpu GPU compute |
| `remote` | TCP-based distributed execution |
| `remote-tls` | TLS-secured remote execution |
| `tensor` | trueno SIMD tensor integration |
| `checkpoint` | State persistence with trueno-db |

## Toyota Way Principles

- **Heijunka**: Work-stealing levels the processing load
- **Jidoka**: Stop on task failure, automatic retry
- **Muda**: Eliminate idle workers through stealing
