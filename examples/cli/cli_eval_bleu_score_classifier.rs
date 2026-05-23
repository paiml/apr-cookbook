//! # apr eval --metric bleu — Score Classifier
//!
//! `apr eval --metric bleu` returns BLEU-4 in [0, 100]. Operational
//! tiers: < 10 = unusable, 10-25 = comprehensible, 25-40 = good,
//! 40-60 = excellent (human-tier for many MT pairs), 60+ = suspect
//! (often dataset contamination). This recipe builds the classifier.
//!
//! Demonstrates the **EVAL.4** recipe for PMAT-112 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EVAL-001 + Papineni et al. 2002 (BLEU)
//!
//! Run with: cargo run --example cli_eval_bleu_score_classifier
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub enum BleuTier {
    Unusable,
    Comprehensible,
    Good,
    Excellent,
    SuspectContamination,
    OutOfRange,
}

pub fn classify(score: f64) -> BleuTier {
    if !(0.0..=100.0).contains(&score) {
        return BleuTier::OutOfRange;
    }
    if score < 10.0 {
        BleuTier::Unusable
    } else if score < 25.0 {
        BleuTier::Comprehensible
    } else if score < 40.0 {
        BleuTier::Good
    } else if score < 60.0 {
        BleuTier::Excellent
    } else {
        BleuTier::SuspectContamination
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_bleu_score_classifier")?;

    for s in [0.0, 8.5, 18.0, 32.0, 45.5, 72.0, -1.0, 110.0] {
        println!("BLEU={s:>6.2}  →  {:?}", classify(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_10_unusable() {
        assert_eq!(classify(0.0), BleuTier::Unusable);
        assert_eq!(classify(9.99), BleuTier::Unusable);
    }

    #[test]
    fn ten_to_25_comprehensible() {
        assert_eq!(classify(10.0), BleuTier::Comprehensible);
        assert_eq!(classify(24.99), BleuTier::Comprehensible);
    }

    #[test]
    fn twentyfive_to_40_good() {
        assert_eq!(classify(25.0), BleuTier::Good);
        assert_eq!(classify(39.99), BleuTier::Good);
    }

    #[test]
    fn forty_to_60_excellent() {
        assert_eq!(classify(40.0), BleuTier::Excellent);
        assert_eq!(classify(59.99), BleuTier::Excellent);
    }

    #[test]
    fn over_60_suspect_contamination() {
        // BLEU > 60 on standard MT pairs typically signals overlap with
        // training data (e.g., bitext leak or sentence duplication).
        assert_eq!(classify(60.0), BleuTier::SuspectContamination);
        assert_eq!(classify(95.0), BleuTier::SuspectContamination);
    }

    #[test]
    fn negative_out_of_range() {
        assert_eq!(classify(-0.01), BleuTier::OutOfRange);
        assert_eq!(classify(-100.0), BleuTier::OutOfRange);
    }

    #[test]
    fn over_100_out_of_range() {
        assert_eq!(classify(100.01), BleuTier::OutOfRange);
        assert_eq!(classify(1000.0), BleuTier::OutOfRange);
    }

    #[test]
    fn perfect_100_classified_as_suspect() {
        // 100.0 is inside [0, 100] so it passes the range check, then
        // falls into the SuspectContamination tier (well above 60).
        assert_eq!(classify(100.0), BleuTier::SuspectContamination);
    }
}
