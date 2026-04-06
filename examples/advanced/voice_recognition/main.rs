#![allow(unused_imports)]
//! # Demo H: Voice Recognition Pipeline
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Simulates a speech-to-text pipeline with mel spectrogram processing
//! and CTC-style decoding. Demonstrates audio preprocessing concepts.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic silence/noise detection
//! - **Heijunka**: Consistent latency regardless of audio length
//! - **Genchi Genbutsu**: Process real audio patterns
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. arXiv:2212.04356

use std::f32::consts::PI;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo H: Voice Recognition Pipeline ===\n");

    let mut generator = AudioGenerator::new(42);

    // Generate different audio types
    let silence = AudioGenerator::silence(1.0, SAMPLE_RATE);
    let sine = generator.sine_wave(440.0, 1.0, SAMPLE_RATE);
    let speech = generator.speech_like(2.0, SAMPLE_RATE);

    let recognizer = VoiceRecognizer::new(42);

    println!("--- Processing Silence ---");
    let result = recognizer.recognize(&silence);
    println!(
        "Text: \"{}\" (conf: {:.2})\n",
        result.text, result.confidence
    );

    println!("--- Processing Sine Wave (440 Hz) ---");
    let result = recognizer.recognize(&sine);
    println!(
        "Text: \"{}\" (conf: {:.2})\n",
        result.text, result.confidence
    );

    println!("--- Processing Speech-like Signal ---");
    let result = recognizer.recognize(&speech);
    println!("Text: \"{}\" (conf: {:.2})", result.text, result.confidence);
    println!("Words: {}", result.word_count());

    println!("\n--- Mel Spectrogram Stats ---");
    let mel = MelSpectrogram::from_audio(&speech);
    println!("Frames: {}", mel.num_frames);
    println!("Duration: {:.2}s", mel.duration());

    println!("\n=== Demo H Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_samples_new() {
        let audio = AudioSamples::new(vec![0.0; 100], 16000);
        assert_eq!(audio.samples.len(), 100);
        assert_eq!(audio.sample_rate, 16000);
    }

    #[test]
    fn test_audio_duration() {
        let audio = AudioSamples::new(vec![0.0; 16000], 16000);
        assert!((audio.duration() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_rms_silence() {
        let audio = AudioSamples::new(vec![0.0; 100], 16000);
        assert!((audio.rms() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_rms_signal() {
        let audio = AudioSamples::new(vec![1.0; 100], 16000);
        assert!((audio.rms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_is_silence() {
        let silence = AudioSamples::new(vec![0.0; 100], 16000);
        assert!(silence.is_silence(0.01));

        let loud = AudioSamples::new(vec![0.5; 100], 16000);
        assert!(!loud.is_silence(0.01));
    }

    #[test]
    fn test_audio_normalize() {
        let mut audio = AudioSamples::new(vec![0.0, 0.5, 1.0, -0.5], 16000);
        audio.normalize();
        assert!((audio.samples[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_resample() {
        let audio = AudioSamples::new(vec![0.0; 16000], 16000);
        let resampled = audio.resample(8000);
        assert_eq!(resampled.sample_rate, 8000);
        assert_eq!(resampled.samples.len(), 8000);
    }

    #[test]
    fn test_mel_spectrogram_new() {
        let mel = MelSpectrogram::new();
        assert_eq!(mel.num_frames, 0);
    }

    #[test]
    fn test_mel_spectrogram_add_frame() {
        let mut mel = MelSpectrogram::new();
        mel.add_frame(&[0.0; MEL_BINS]);
        assert_eq!(mel.num_frames, 1);
    }

    #[test]
    fn test_mel_from_audio() {
        let audio = AudioSamples::new(vec![0.1; 2000], SAMPLE_RATE);
        let mel = MelSpectrogram::from_audio(&audio);
        assert!(mel.num_frames > 0);
    }

    #[test]
    fn test_audio_processor_new() {
        let processor = AudioProcessor::new();
        assert_eq!(processor.window.len(), FFT_SIZE);
    }

    #[test]
    fn test_hz_to_mel() {
        assert!((hz_to_mel(0.0) - 0.0).abs() < 0.1);
        assert!(hz_to_mel(1000.0) > 0.0);
    }

    #[test]
    fn test_mel_to_hz() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((back - hz).abs() < 1.0);
    }

    #[test]
    fn test_ctc_decoder_new() {
        let decoder = CTCDecoder::new();
        assert!((decoder.min_prob - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ctc_decoder_empty() {
        let decoder = CTCDecoder::new();
        let result = decoder.decode(&[]);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_ctc_decoder_blank_only() {
        let decoder = CTCDecoder::new();
        let mut frame = [0.0_f32; VOCAB_SIZE];
        frame[BLANK_TOKEN] = 1.0;
        let result = decoder.decode(&[frame]);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_ctc_decoder_character() {
        let decoder = CTCDecoder::new();
        let mut frame = [0.0_f32; VOCAB_SIZE];
        frame[1] = 1.0; // 'a'
        let result = decoder.decode(&[frame]);
        assert_eq!(result.text, "a");
    }

    #[test]
    fn test_transcription_new() {
        let t = Transcription::new("hello", 0.9);
        assert_eq!(t.text, "hello");
        assert!((t.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_transcription_word_count() {
        let t = Transcription::new("hello world test", 0.9);
        assert_eq!(t.word_count(), 3);
    }

    #[test]
    fn test_voice_recognizer_new() {
        let _recognizer = VoiceRecognizer::new(42);
        // Just verify it creates successfully
        assert!(true);
    }

    #[test]
    fn test_voice_recognizer_silence() {
        let recognizer = VoiceRecognizer::new(42);
        let silence = AudioSamples::new(vec![0.0; 16000], 16000);
        let result = recognizer.recognize(&silence);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_audio_generator_sine() {
        let generator = AudioGenerator::new(42);
        let audio = generator.sine_wave(440.0, 1.0, 16000);
        assert_eq!(audio.samples.len(), 16000);
        assert!(!audio.is_silence(0.01));
    }

    #[test]
    fn test_audio_generator_silence() {
        let audio = AudioGenerator::silence(1.0, 16000);
        assert!(audio.is_silence(0.01));
    }

    #[test]
    fn test_audio_generator_noise() {
        let mut generator = AudioGenerator::new(42);
        let audio = generator.white_noise(1.0, 16000);
        assert!(!audio.is_silence(0.01));
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(VOCAB.len(), VOCAB_SIZE);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn prop_audio_duration_positive(len in 100usize..10000, rate in 8000u32..48000) {
            let audio = AudioSamples::new(vec![0.0; len], rate);
            prop_assert!(audio.duration() > 0.0);
        }

        #[test]
        fn prop_rms_non_negative(len in 10usize..1000) {
            let samples: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
            let audio = AudioSamples::new(samples, 16000);
            prop_assert!(audio.rms() >= 0.0);
        }

        #[test]
        fn prop_mel_hz_roundtrip(hz in 20.0f32..8000.0) {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            prop_assert!((back - hz).abs() < 1.0);
        }

        #[test]
        fn prop_resample_changes_length(len in 1000usize..5000, ratio in 1u32..3) {
            let audio = AudioSamples::new(vec![0.0; len], 16000);
            let target = 16000 * ratio;
            let resampled = audio.resample(target);
            let expected_len = (len as f32 * ratio as f32) as usize;
            prop_assert!((resampled.samples.len() as i32 - expected_len as i32).abs() <= 1);
        }

        #[test]
        fn prop_ctc_decode_deterministic(seed in 0u64..1000) {
            let decoder = CTCDecoder::new();
            let mut rng = SimpleRng::new(seed);
            let mut frame = [0.0_f32; VOCAB_SIZE];
            for p in &mut frame {
                *p = rng.next_f32();
            }
            let r1 = decoder.decode(&[frame]);
            let r2 = decoder.decode(&[frame]);
            prop_assert_eq!(r1.text, r2.text);
        }

        #[test]
        fn prop_transcription_word_count(n in 1usize..10) {
            let words: Vec<&str> = vec!["hello", "world", "test", "foo", "bar", "baz", "qux", "abc", "def", "ghi"];
            let text = words[..n].join(" ");
            let t = Transcription::new(&text, 0.9);
            prop_assert_eq!(t.word_count(), n);
        }
    }
}
