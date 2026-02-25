# Voice Activity Detection

Frame-based voice activity detection on a synthetic audio stream using energy, zero-crossing rate, and spectral centroid features. Includes median smoothing and consecutive-frame merging for clean segment boundaries.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Frame-level feature extraction (RMS, ZCR, spectral centroid)
- Threshold-based speech/silence classification
- Median smoothing and segment merging

## Run
```bash
cargo run --example speech_vad
```

## Source
[`examples/speech/speech_vad.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/speech/speech_vad.rs)
