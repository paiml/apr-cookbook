# Whisper Transcription

Transcribe audio files using whisper.apr with APR v2 model format.

## Example

```bash
cargo run --example whisper_transcribe
```

## Code

```rust
//! Whisper Transcription Example
//!
//! Demonstrates audio transcription with whisper.apr.

use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("whisper_transcribe")?;

    // Create whisper model (simulated)
    let model = WhisperModel::new(WhisperConfig {
        size: ModelSize::Small,
        quantization: Quantization::Int8,
        language: None, // Auto-detect
    });

    println!("Model: {} ({})", model.name(), model.size_category());
    println!("Quantization: {:?}", model.quantization());

    // Transcribe audio
    let audio = generate_test_audio(16000, 3.0); // 3 seconds at 16kHz
    let result = model.transcribe(&audio, TranscribeOptions::default())?;

    println!("\n=== Transcription ===");
    println!("Text: {}", result.text);
    println!("Language: {} ({:.1}% confidence)",
        result.language, result.language_confidence * 100.0);

    if !result.segments.is_empty() {
        println!("\n=== Segments ===");
        for seg in &result.segments {
            println!("[{:.2}s - {:.2}s] {}", seg.start, seg.end, seg.text);
        }
    }

    ctx.record_float_metric("confidence", result.language_confidence as f64);
    ctx.report()?;

    Ok(())
}
```

## Key Features

### Language Detection

Whisper automatically detects the spoken language:

```rust
let result = model.transcribe(&audio, TranscribeOptions::default())?;
println!("Detected: {} ({:.1}%)", result.language, result.confidence * 100.0);
```

### Timestamps

Enable word-level timestamps:

```rust
let options = TranscribeOptions {
    with_timestamps: true,
    word_timestamps: true,
    ..Default::default()
};

let result = model.transcribe(&audio, options)?;
for word in &result.words {
    println!("[{:.2}s] {}", word.start, word.text);
}
```

### Quantized Models

Use Int4/Int8 quantization for smaller models:

```rust
// Load Int8 quantized model (4x smaller)
const MODEL: &[u8] = include_bytes!("whisper-small-int8.apr");
let model = WhisperModel::from_apr_bytes(MODEL)?;

// Load Int4 quantized model (8x smaller)
const MODEL_Q4: &[u8] = include_bytes!("whisper-small-int4.apr");
let model_q4 = WhisperModel::from_apr_bytes(MODEL_Q4)?;
```

## Falsifiable Claims

**F5**: whisper.apr Int8 model achieves WER <10% on LibriSpeech test-clean.

```rust
#[test]
fn f5_speech_recognition_wer() {
    let wer = calculate_wer(reference, hypothesis);
    assert!(wer < 0.12, "FALSIFIED: WER {:.2}% > 12%", wer * 100.0);
}
```

## Tests

```rust
#[test]
fn test_language_detection() {
    let model = WhisperModel::new(Default::default());
    let audio = generate_english_audio();

    let result = model.transcribe(&audio, Default::default()).unwrap();
    assert_eq!(result.language, "en");
}

#[test]
fn test_timestamp_consistency() {
    let model = WhisperModel::new(Default::default());
    let audio = generate_test_audio(16000, 5.0);

    let result = model.transcribe(&audio, TranscribeOptions {
        with_timestamps: true,
        ..Default::default()
    }).unwrap();

    // Timestamps should be monotonically increasing
    for window in result.segments.windows(2) {
        assert!(window[0].end <= window[1].start);
    }
}
```
