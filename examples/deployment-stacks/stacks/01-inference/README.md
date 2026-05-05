# Stack 01: Inference

Single-machine model serving with **realizar**.

## What it deploys

- GPU driver + CUDA toolkit
- `realizar` binary via `cargo install`
- Model download with BLAKE3 verification
- Systemd service unit
- Firewall rule (port 8080)
- Health check cron (every 5 minutes)

## Usage

```bash
# Validate
forjar validate -f stacks/01-inference/forjar.yaml

# Plan (dry-run)
forjar plan -f stacks/01-inference/forjar.yaml

# Apply
forjar apply -f stacks/01-inference/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model_path` | `/opt/models/llama-2-7b.gguf` | Path to model file |
| `serve_port` | `8080` | HTTP listen port |
| `workers` | `1` | Inference worker count |

## Customization

Override params at apply time:

```bash
forjar apply -f stacks/01-inference/forjar.yaml \
  --set model_path=/opt/models/mistral-7b.gguf \
  --set serve_port=9090
```
