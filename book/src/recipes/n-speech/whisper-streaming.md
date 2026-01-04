# Streaming ASR

Real-time speech recognition with whisper.apr streaming API.

## Example

```bash
cargo run --example whisper_streaming
```

## Code

```rust
//! Streaming ASR Example
//!
//! Demonstrates real-time speech recognition with whisper.apr.

use apr_cookbook::prelude::*;
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("whisper_streaming")?;

    // Create streaming transcriber
    let model = WhisperModel::new(WhisperConfig {
        size: ModelSize::Tiny, // Use tiny for low latency
        quantization: Quantization::Int4,
        ..Default::default()
    });

    let mut streamer = StreamingTranscriber::new(model);

    println!("Streaming transcription (simulated)");
    println!("Processing audio chunks...\n");

    // Simulate streaming audio chunks
    let chunk_size = 4096; // 256ms at 16kHz
    let total_samples = 16000 * 3; // 3 seconds

    for chunk_start in (0..total_samples).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(total_samples);
        let chunk: Vec<f32> = (chunk_start..chunk_end)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();

        if let Some(partial) = streamer.process_chunk(&chunk)? {
            print!("\r{}", partial.text);
            io::Write::flush(&mut io::stdout())?;
        }
    }

    // Finalize transcription
    let final_result = streamer.finalize()?;
    println!("\n\nFinal: {}", final_result.text);

    ctx.record_float_metric("latency_ms", streamer.avg_latency_ms() as f64);
    ctx.report()?;

    Ok(())
}
```

## Key Features

### Low Latency Processing

Streaming mode processes audio in chunks for real-time feedback:

```rust
let mut streamer = StreamingTranscriber::new(model);

// Process 256ms chunks
for chunk in audio_stream {
    if let Some(partial) = streamer.process_chunk(&chunk)? {
        // Update UI with partial transcription
        display_partial(&partial.text);
    }
}

// Get final result
let final_result = streamer.finalize()?;
```

### Voice Activity Detection

Only process chunks with speech:

```rust
let mut streamer = StreamingTranscriber::new(model)
    .with_vad(VadConfig {
        threshold: 0.5,
        min_speech_duration_ms: 250,
        min_silence_duration_ms: 500,
    });
```

### Buffer Management

Configure buffering for latency vs accuracy tradeoff:

```rust
let mut streamer = StreamingTranscriber::new(model)
    .with_buffer_size(8192)      // ~500ms buffer
    .with_overlap(1024)          // 64ms overlap
    .with_max_pending_chunks(4); // Process up to 4 chunks
```

## Performance

| Model | Chunk Size | Latency | RTF |
|-------|------------|---------|-----|
| Tiny | 256ms | ~50ms | 0.2x |
| Base | 256ms | ~100ms | 0.4x |
| Small | 512ms | ~200ms | 0.4x |

RTF = Real-Time Factor (lower is faster)

## Tests

```rust
#[test]
fn test_streaming_produces_output() {
    let model = WhisperModel::new(Default::default());
    let mut streamer = StreamingTranscriber::new(model);

    let audio = generate_test_audio(16000, 2.0);
    let chunks: Vec<_> = audio.chunks(4096).collect();

    let mut saw_output = false;
    for chunk in chunks {
        if streamer.process_chunk(chunk).unwrap().is_some() {
            saw_output = true;
        }
    }

    let final_result = streamer.finalize().unwrap();
    assert!(!final_result.text.is_empty() || saw_output);
}

#[test]
fn test_streaming_latency() {
    let model = WhisperModel::new(WhisperConfig {
        size: ModelSize::Tiny,
        ..Default::default()
    });
    let mut streamer = StreamingTranscriber::new(model);

    // Process one chunk and measure latency
    let chunk = vec![0.0f32; 4096];
    let start = std::time::Instant::now();
    let _ = streamer.process_chunk(&chunk);
    let latency = start.elapsed();

    // Should be under 100ms for tiny model
    assert!(latency.as_millis() < 100);
}
```
