# Stack 02: Training

GPU training pipeline with **entrenar**.

## What it deploys

- GPU driver + CUDA toolkit
- `entrenar` binary via `cargo install`
- Training configuration (learning rate, epochs, LoRA rank)
- Dataset + output + checkpoint directories
- Systemd oneshot unit for batch training

## Usage

```bash
forjar validate -f stacks/02-training/forjar.yaml
forjar plan -f stacks/02-training/forjar.yaml
forjar apply -f stacks/02-training/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `dataset_path` | `/opt/datasets/training-data` | Training dataset directory |
| `output_dir` | `/opt/entrenar/output` | Model output directory |
| `learning_rate` | `2e-5` | Learning rate |
| `epochs` | `3` | Number of epochs |
| `batch_size` | `8` | Batch size |
| `lora_rank` | `16` | LoRA rank (0 = full fine-tune) |
