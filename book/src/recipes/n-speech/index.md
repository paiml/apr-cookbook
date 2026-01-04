# Category N: Speech Recognition

This category covers speech recognition using whisper.apr, a pure Rust implementation of OpenAI's Whisper model.

## Recipes

| Recipe | Description |
|--------|-------------|
| [Whisper Transcription](./whisper-transcribe.md) | Transcribe audio files to text |
| [Streaming ASR](./whisper-streaming.md) | Real-time speech recognition |

## Key Concepts

### whisper.apr

A pure Rust implementation of Whisper designed for:
- **WASM-first**: Runs in browsers without server
- **Int4/Int8 Quantization**: 4x smaller models
- **Streaming**: Real-time transcription
- **APR v2 Format**: Fast loading with LZ4 compression

### Model Sizes

| Model | Parameters | Size (Int8) | WER |
|-------|------------|-------------|-----|
| Tiny | 39M | 40MB | ~15% |
| Base | 74M | 75MB | ~12% |
| Small | 244M | 250MB | ~10% |
| Medium | 769M | 800MB | ~8% |

## Stack Integration

```rust
use whisper_apr::{WhisperModel, Transcriber, TranscribeOptions};

// Load quantized model from APR v2
const MODEL: &[u8] = include_bytes!("whisper-small-int8.apr");
let model = WhisperModel::from_apr_bytes(MODEL)?;

// Create transcriber
let transcriber = Transcriber::new(model);

// Transcribe audio
let result = transcriber.transcribe_file("audio.wav", TranscribeOptions::default())?;
println!("Text: {}", result.text);
println!("Language: {} ({:.1}%)", result.language, result.confidence * 100.0);
```

## Falsifiable Claims

| Claim | Metric | Threshold |
|-------|--------|-----------|
| F5 | Word Error Rate | <10% on LibriSpeech |

## Toyota Way Principles

- **Jidoka**: Stop on unrecognizable audio
- **Muda**: Quantization eliminates size waste
- **Heijunka**: Streaming levels processing load
