# Stack 07: Data Pipeline

End-to-end pipeline: **alimentar** ingest → **entrenar** train → **realizar** serve.

## What it deploys

Three sequential stages on a single GPU machine:

1. **alimentar**: Ingests raw data, preprocesses, outputs JSONL
2. **entrenar**: Fine-tunes model on ingested data with LoRA
3. **realizar**: Serves the trained model

## Pipeline Flow

```
Raw Data → alimentar (ingest) → JSONL Dataset → entrenar (train) → Model → realizar (serve)
```

## Usage

```bash
forjar validate -f stacks/07-data-pipeline/forjar.yaml
forjar plan -f stacks/07-data-pipeline/forjar.yaml
forjar apply -f stacks/07-data-pipeline/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `source_dir` | `/opt/raw-data` | Raw data source directory |
| `dataset_path` | `/opt/alimentar/output` | Processed dataset output |
| `model_output` | `/opt/entrenar/output` | Trained model output |
