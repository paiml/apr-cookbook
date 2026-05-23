//! # Speech Audio Resampler Picker
//!
//! Whisper expects 16 kHz mono. Common input rates: 8 kHz (telephony),
//! 16 kHz (already OK), 22.05 kHz (legacy), 44.1 kHz (CD), 48 kHz
//! (consumer DAC). Resampling adds anti-alias filter cost. This recipe
//! picks the upsample/downsample factor + filter quality.
//!
//! Demonstrates the **SPEECH.4** recipe for PMAT-140 (speech round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Whisper paper § input feature pipeline (16 kHz mono).
//!
//! Run with: cargo run --example speech_audio_resampler
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const TARGET_HZ: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterQuality {
    Linear,
    Sinc,
    SincHighQuality,
}

#[derive(Debug, PartialEq)]
pub enum ResampleVerdict {
    NoOp,
    Upsample {
        ratio: u32,
        filter: FilterQuality,
    },
    Downsample {
        numer: u32,
        denom: u32,
        filter: FilterQuality,
    },
    InvalidRate,
}

pub fn pick(input_hz: u32) -> ResampleVerdict {
    if input_hz == 0 || input_hz > 192_000 {
        return ResampleVerdict::InvalidRate;
    }
    if input_hz == TARGET_HZ {
        return ResampleVerdict::NoOp;
    }
    if input_hz < TARGET_HZ && TARGET_HZ % input_hz == 0 {
        let ratio = TARGET_HZ / input_hz;
        return ResampleVerdict::Upsample {
            ratio,
            filter: FilterQuality::Sinc,
        };
    }
    let g = gcd(input_hz, TARGET_HZ);
    let numer = TARGET_HZ / g;
    let denom = input_hz / g;
    let filter = if denom == 1 || input_hz <= 22_050 {
        FilterQuality::Sinc
    } else {
        FilterQuality::SincHighQuality
    };
    ResampleVerdict::Downsample {
        numer,
        denom,
        filter,
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_audio_resampler")?;

    for hz in [8_000u32, 16_000, 22_050, 44_100, 48_000, 192_001, 0] {
        println!("{hz} Hz: {:?}", pick(hz));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn at_target_no_op() {
        assert_eq!(pick(16_000), ResampleVerdict::NoOp);
    }

    #[test]
    fn telephony_upsamples() {
        // 16000 / 8000 = 2.
        let v = pick(8_000);
        assert!(matches!(v, ResampleVerdict::Upsample { ratio: 2, .. }));
    }

    #[test]
    fn cd_downsamples() {
        // 44100 → 16000: gcd(44100, 16000) = 100; 160:441.
        let v = pick(44_100);
        if let ResampleVerdict::Downsample { numer, denom, .. } = v {
            assert_eq!(numer, 160);
            assert_eq!(denom, 441);
        }
    }

    #[test]
    fn dac_downsamples_evenly() {
        // 48000 → 16000: 1:3.
        let v = pick(48_000);
        if let ResampleVerdict::Downsample { numer, denom, .. } = v {
            assert_eq!(numer, 1);
            assert_eq!(denom, 3);
        }
    }

    #[test]
    fn invalid_zero_rate() {
        assert_eq!(pick(0), ResampleVerdict::InvalidRate);
    }

    #[test]
    fn invalid_excessive_rate() {
        assert_eq!(pick(192_001), ResampleVerdict::InvalidRate);
    }

    #[test]
    fn high_freq_uses_high_quality() {
        let v = pick(44_100);
        if let ResampleVerdict::Downsample { filter, .. } = v {
            assert_eq!(filter, FilterQuality::SincHighQuality);
        }
    }

    #[test]
    fn low_freq_uses_regular_sinc() {
        // 22050 → 16000: gcd 50; 320:441.
        let v = pick(22_050);
        if let ResampleVerdict::Downsample { filter, .. } = v {
            assert_eq!(filter, FilterQuality::Sinc);
        }
    }

    #[test]
    fn upsample_2x_factor() {
        // 8 kHz → 16 kHz, ratio = 2.
        let v = pick(8_000);
        if let ResampleVerdict::Upsample { ratio, .. } = v {
            assert_eq!(ratio, 2);
        }
    }

    #[test]
    fn rates_at_max_allowed() {
        // 192000 Hz is the max allowed.
        let v = pick(192_000);
        assert!(matches!(v, ResampleVerdict::Downsample { .. }));
    }
}
