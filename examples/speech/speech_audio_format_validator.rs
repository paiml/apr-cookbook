//! # Speech Audio Format Validator
//!
//! Whisper accepts 16-kHz mono WAV/FLAC/MP3 input. This recipe builds
//! the format detector (RIFF/fLaC/ID3 magic bytes) + sample-rate
//! envelope + channel-count guard. Wrong rate → resample required;
//! wrong channels → downmix required.
//!
//! Demonstrates the **SPEECH.2** recipe for PMAT-123 (speech coverage —
//! closing F-invariant gap from 1 → 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). arXiv:2212.04356.
//!
//! Run with: cargo run --example speech_audio_format_validator
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum FormatVerdict {
    Ok { format: AudioFormat },
    HeaderTooShort,
    UnknownFormat,
}

pub fn detect_format(magic: &[u8]) -> FormatVerdict {
    if magic.len() < 4 {
        return FormatVerdict::HeaderTooShort;
    }
    if &magic[..4] == b"RIFF" && magic.len() >= 12 && &magic[8..12] == b"WAVE" {
        return FormatVerdict::Ok {
            format: AudioFormat::Wav,
        };
    }
    if &magic[..4] == b"fLaC" {
        return FormatVerdict::Ok {
            format: AudioFormat::Flac,
        };
    }
    if &magic[..3] == b"ID3" || (magic[0] == 0xFF && (magic[1] & 0xE0) == 0xE0) {
        return FormatVerdict::Ok {
            format: AudioFormat::Mp3,
        };
    }
    FormatVerdict::UnknownFormat
}

#[derive(Debug, PartialEq)]
pub enum AudioVerdict {
    NativelySupported,
    NeedsResample { current_rate: u32, target_rate: u32 },
    NeedsDownmix { channels: u8 },
    InvalidShape,
}

const TARGET_RATE: u32 = 16_000;

pub fn classify_audio(sample_rate: u32, channels: u8) -> AudioVerdict {
    if sample_rate == 0 || channels == 0 {
        return AudioVerdict::InvalidShape;
    }
    if channels > 1 {
        return AudioVerdict::NeedsDownmix { channels };
    }
    if sample_rate == TARGET_RATE {
        return AudioVerdict::NativelySupported;
    }
    AudioVerdict::NeedsResample {
        current_rate: sample_rate,
        target_rate: TARGET_RATE,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_audio_format_validator")?;

    let wav = b"RIFF\x00\x00\x00\x00WAVEfmt ";
    let flac = b"fLaC\x00\x00\x00\x22";
    let mp3 = b"ID3\x04\x00\x00\x00\x00";
    println!("WAV:  {:?}", detect_format(wav));
    println!("FLAC: {:?}", detect_format(flac));
    println!("MP3:  {:?}", detect_format(mp3));
    println!("?:    {:?}", detect_format(b"XYZ "));

    for (rate, ch) in [(16_000, 1u8), (44_100, 1), (16_000, 2), (0, 1)] {
        println!("rate={rate} ch={ch}  →  {:?}", classify_audio(rate, ch));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn wav_magic_detected() {
        let v = detect_format(b"RIFF\x00\x00\x00\x00WAVEfmt ");
        assert!(matches!(
            v,
            FormatVerdict::Ok {
                format: AudioFormat::Wav
            }
        ));
    }

    #[test]
    fn flac_magic_detected() {
        let v = detect_format(b"fLaC\x00\x00\x00\x22");
        assert!(matches!(
            v,
            FormatVerdict::Ok {
                format: AudioFormat::Flac
            }
        ));
    }

    #[test]
    fn mp3_id3_magic_detected() {
        let v = detect_format(b"ID3\x04\x00\x00\x00\x00");
        assert!(matches!(
            v,
            FormatVerdict::Ok {
                format: AudioFormat::Mp3
            }
        ));
    }

    #[test]
    fn mp3_frame_sync_detected() {
        // MP3 without ID3: frame starts with 0xFFF (sync word).
        let v = detect_format(&[0xFF, 0xFB, 0x90, 0x44]);
        assert!(matches!(
            v,
            FormatVerdict::Ok {
                format: AudioFormat::Mp3
            }
        ));
    }

    #[test]
    fn truncated_header_rejected() {
        assert_eq!(detect_format(b"RI"), FormatVerdict::HeaderTooShort);
    }

    #[test]
    fn riff_without_wave_rejected() {
        // RIFF could be AVI (RIFF + AVI ); check requires WAVE form.
        let v = detect_format(b"RIFF\x00\x00\x00\x00AVI ");
        assert!(matches!(v, FormatVerdict::UnknownFormat));
    }

    #[test]
    fn natively_supported_at_16khz_mono() {
        assert_eq!(classify_audio(16_000, 1), AudioVerdict::NativelySupported);
    }

    #[test]
    fn needs_resample_at_44100() {
        let v = classify_audio(44_100, 1);
        assert!(matches!(v, AudioVerdict::NeedsResample { .. }));
    }

    #[test]
    fn needs_downmix_for_stereo() {
        let v = classify_audio(16_000, 2);
        assert!(matches!(v, AudioVerdict::NeedsDownmix { channels: 2 }));
    }

    #[test]
    fn zero_rate_or_channels_invalid() {
        assert_eq!(classify_audio(0, 1), AudioVerdict::InvalidShape);
        assert_eq!(classify_audio(16_000, 0), AudioVerdict::InvalidShape);
    }

    #[test]
    fn downmix_takes_priority_over_resample() {
        // Stereo at non-target rate: downmix first (cheaper), then resample.
        let v = classify_audio(44_100, 2);
        assert!(matches!(v, AudioVerdict::NeedsDownmix { .. }));
    }
}
