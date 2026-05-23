# Stack 04: Speech

Automatic speech recognition with **whisper-apr**.

## What it deploys

- GPU driver + CUDA toolkit
- `whisper-apr` binary via `cargo install`
- Whisper model download
- ASR service (port 8095)
- Health check cron

## Usage

```bash
forjar validate -f stacks/04-speech/forjar.yaml
forjar plan -f stacks/04-speech/forjar.yaml
forjar apply -f stacks/04-speech/forjar.yaml
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model_size` | `base` | Whisper model (tiny/base/small/medium/large) |
| `language` | `en` | Default recognition language |
| `asr_port` | `8095` | ASR API port |
