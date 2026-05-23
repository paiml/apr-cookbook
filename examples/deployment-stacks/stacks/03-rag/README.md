# Stack 03: RAG

Retrieval-augmented generation with **trueno-db** + **trueno-rag** + **realizar**.

## What it deploys

- **trueno-db**: Vector/analytics database for embeddings (port 5433)
- **trueno-rag**: Embedding + retrieval pipeline (port 8090)
- **realizar**: Generation model server (port 8080)

## Architecture

```
Documents → trueno-rag (embed + index) → trueno-db (vector storage)
Query → trueno-rag (retrieve) → realizar (generate) → Response
```

## Usage

```bash
forjar validate -f stacks/03-rag/forjar.yaml
forjar plan -f stacks/03-rag/forjar.yaml
forjar apply -f stacks/03-rag/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model_path` | `/opt/models/llama-2-7b.gguf` | Generation model path |
| `embedding_dim` | `384` | Embedding vector dimension |
| `chunk_size` | `512` | Document chunk size (tokens) |
