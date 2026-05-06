//! # Speech Language-ID Confidence Picker
//!
//! Whisper's language detector returns top-K language probabilities
//! per audio chunk. Decision rules:
//!   top_prob >= 0.95          → ConfidentLang
//!   0.50 ≤ top_prob < 0.95    → AmbiguousFallbackToConfig
//!   top_prob < 0.50           → UnknownDefaultMultilingual
//!
//! Plus: top-2 margin gates (small margin → still ambiguous even if
//! top is high).
//!
//! Demonstrates the **SPEECH.6** recipe for PMAT-140 (speech round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Whisper § 4.5 Language identification.
//!
//! Run with: cargo run --example speech_language_id_confidence
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const HIGH_CONF_THRESHOLD: f64 = 0.95;
const MIN_USEFUL_THRESHOLD: f64 = 0.50;
const MIN_TOP2_MARGIN: f64 = 0.10;

#[derive(Debug, PartialEq)]
pub enum LangVerdict {
    ConfidentLang { iso: String, prob: f64 },
    AmbiguousFallback { iso: String, prob: f64 },
    UnknownMultilingual,
    InvalidProbabilities,
    EmptyCandidates,
}

pub fn pick(candidates: &[(&str, f64)]) -> LangVerdict {
    if candidates.is_empty() {
        return LangVerdict::EmptyCandidates;
    }
    if candidates
        .iter()
        .any(|(_, p)| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return LangVerdict::InvalidProbabilities;
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (iso, top_prob) = sorted[0];
    let second_prob = sorted.get(1).map_or(0.0, |x| x.1);
    let margin = top_prob - second_prob;
    if top_prob >= HIGH_CONF_THRESHOLD && margin >= MIN_TOP2_MARGIN {
        return LangVerdict::ConfidentLang {
            iso: iso.to_string(),
            prob: top_prob,
        };
    }
    if top_prob >= MIN_USEFUL_THRESHOLD {
        return LangVerdict::AmbiguousFallback {
            iso: iso.to_string(),
            prob: top_prob,
        };
    }
    LangVerdict::UnknownMultilingual
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_language_id_confidence")?;

    println!(
        "confident: {:?}",
        pick(&[("en", 0.98), ("de", 0.01), ("fr", 0.01)])
    );
    println!(
        "ambiguous high but close: {:?}",
        pick(&[("en", 0.96), ("de", 0.93), ("fr", 0.01)])
    );
    println!(
        "low conf: {:?}",
        pick(&[("en", 0.4), ("de", 0.4), ("fr", 0.2)])
    );
    println!(
        "very low: {:?}",
        pick(&[("en", 0.2), ("de", 0.2), ("fr", 0.2)])
    );
    println!("empty: {:?}", pick(&[]));
    println!("invalid: {:?}", pick(&[("en", 1.5)]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn very_confident_picks_confident() {
        let v = pick(&[("en", 0.98), ("de", 0.01), ("fr", 0.01)]);
        assert!(matches!(v, LangVerdict::ConfidentLang { .. }));
    }

    #[test]
    fn close_top2_demotes_to_ambiguous() {
        // Top 0.96 ≥ 0.95 BUT margin 0.03 < 0.10 → Ambiguous.
        let v = pick(&[("en", 0.96), ("de", 0.93), ("fr", 0.01)]);
        assert!(matches!(v, LangVerdict::AmbiguousFallback { .. }));
    }

    #[test]
    fn medium_conf_ambiguous() {
        let v = pick(&[("en", 0.7), ("de", 0.2), ("fr", 0.1)]);
        assert!(matches!(v, LangVerdict::AmbiguousFallback { .. }));
    }

    #[test]
    fn low_conf_unknown() {
        let v = pick(&[("en", 0.3), ("de", 0.3), ("fr", 0.4)]);
        assert!(matches!(v, LangVerdict::UnknownMultilingual));
    }

    #[test]
    fn empty_candidates_rejected() {
        assert_eq!(pick(&[]), LangVerdict::EmptyCandidates);
    }

    #[test]
    fn invalid_prob_above_one_rejected() {
        let v = pick(&[("en", 1.5)]);
        assert_eq!(v, LangVerdict::InvalidProbabilities);
    }

    #[test]
    fn nan_prob_rejected() {
        let v = pick(&[("en", f64::NAN)]);
        assert_eq!(v, LangVerdict::InvalidProbabilities);
    }

    #[test]
    fn at_high_conf_threshold_with_big_margin_confident() {
        let v = pick(&[("en", 0.95), ("de", 0.04), ("fr", 0.01)]);
        assert!(matches!(v, LangVerdict::ConfidentLang { .. }));
    }

    #[test]
    fn just_below_high_threshold_ambiguous() {
        let v = pick(&[("en", 0.94), ("de", 0.05), ("fr", 0.01)]);
        assert!(matches!(v, LangVerdict::AmbiguousFallback { .. }));
    }

    #[test]
    fn iso_code_returned_in_verdict() {
        if let LangVerdict::ConfidentLang { iso, .. } =
            pick(&[("en", 0.98), ("de", 0.01), ("fr", 0.01)])
        {
            assert_eq!(iso, "en");
        }
    }

    #[test]
    fn single_candidate_high_conf_uses_zero_margin() {
        // Only one candidate → second_prob = 0; margin = top.
        let v = pick(&[("en", 0.99)]);
        assert!(matches!(v, LangVerdict::ConfidentLang { .. }));
    }
}
