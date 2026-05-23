# Stack 05: Distributed Inference

Multi-node inference with **repartir** + **realizar**.

## What it deploys

- **gpu-node**: realizar model server with GPU acceleration
- **worker-node**: repartir distributed execution worker
- Fleet coordination config linking workers to inference

## Architecture

```
Client → realizar (gpu-node:8080) → repartir-worker (worker-node:9000)
                                  → repartir-worker (worker-node-2:9000)
```

## Usage

```bash
forjar validate -f stacks/05-distributed-inference/forjar.yaml
forjar plan -f stacks/05-distributed-inference/forjar.yaml
forjar apply -f stacks/05-distributed-inference/forjar.yaml
```

## Scaling

Add more worker nodes by duplicating the `worker-node` machine entry
and adding corresponding recipe resources.

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model_path` | `/opt/models/llama-2-70b.gguf` | Model path (larger models benefit from distribution) |
