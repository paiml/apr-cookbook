# Stack 06: Full Stack

Complete sovereign AI lab — all PAIML components in one deployment.

## What it deploys

| Machine | Components | Ports |
|---------|-----------|-------|
| gpu-box | realizar (inference) + entrenar (training) | 8080 |
| rag-box | trueno-db + trueno-rag + whisper-apr | 5433, 8090, 8095 |
| worker-box | repartir-worker | 9000 |
| monitor-box | renacer + Grafana + Jaeger + pacha | 3000, 16686, 8070 |

## Architecture

```
                    ┌─────────────┐
                    │ monitor-box │
                    │  Grafana    │
                    │  Jaeger     │
                    │  pacha      │
                    └──────┬──────┘
                           │ observability
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
   │   gpu-box   │  │  rag-box   │  │ worker-box │
   │  realizar   │  │ trueno-db  │  │  repartir  │
   │  entrenar   │  │ trueno-rag │  │            │
   │             │  │ whisper-apr│  │            │
   └─────────────┘  └────────────┘  └────────────┘
```

## Usage

```bash
forjar validate -f stacks/06-full-stack/forjar.yaml
forjar plan -f stacks/06-full-stack/forjar.yaml
forjar apply -f stacks/06-full-stack/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model_path` | `/opt/models/llama-2-7b.gguf` | Inference model path |
| `dataset_path` | `/opt/datasets/training-data` | Training dataset directory |
