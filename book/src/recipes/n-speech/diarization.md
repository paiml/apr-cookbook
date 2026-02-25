# Speaker Diarization

Identifies "who spoke when" in a multi-speaker audio stream using simplified speaker embeddings and k-means clustering. Demonstrates the full pipeline: audio generation, per-frame feature extraction, clustering, and turn merging.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Speaker embedding extraction from audio frames
- K-means clustering for speaker identification
- Turn merging for contiguous speaker segments

## Run
```bash
cargo run --example speech_diarization
```

## Source
[`examples/speech/speech_diarization.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/speech/speech_diarization.rs)
