//! # apr eval --metric — Metric Dispatcher
//!
//! `apr eval --metric <NAME>` accepts a fixed catalog: perplexity,
//! accuracy, top_k_accuracy, bleu, rouge_l, exact_match, mmlu, hellaswag,
//! arc, gsm8k. Some require labels (accuracy, top_k_accuracy), some
//! require references (bleu, rouge_l, exact_match), some are self-scoring
//! (perplexity, mmlu/hellaswag/arc/gsm8k). This recipe builds the
//! capability matrix.
//!
//! Demonstrates the **EVAL.6** recipe for PMAT-112 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EVAL-001 + Hendrycks et al. 2021 (MMLU)
//!
//! Run with: cargo run --example cli_eval_metric_dispatcher
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    SelfScoring,     // perplexity, mmlu, hellaswag, arc, gsm8k
    NeedsLabels,     // accuracy, top_k_accuracy
    NeedsReferences, // bleu, rouge_l, exact_match
}

pub fn classify_metric(name: &str) -> Option<MetricKind> {
    match name {
        "perplexity" | "mmlu" | "hellaswag" | "arc" | "gsm8k" => Some(MetricKind::SelfScoring),
        "accuracy" | "top_k_accuracy" => Some(MetricKind::NeedsLabels),
        "bleu" | "rouge_l" | "exact_match" => Some(MetricKind::NeedsReferences),
        _ => None,
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok,
    UnknownMetric,
    MissingLabels,
    MissingReferences,
}

pub fn dispatch(name: &str, has_labels: bool, has_references: bool) -> DispatchVerdict {
    let Some(kind) = classify_metric(name) else {
        return DispatchVerdict::UnknownMetric;
    };
    match kind {
        MetricKind::SelfScoring => DispatchVerdict::Ok,
        MetricKind::NeedsLabels if !has_labels => DispatchVerdict::MissingLabels,
        MetricKind::NeedsReferences if !has_references => DispatchVerdict::MissingReferences,
        _ => DispatchVerdict::Ok,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_metric_dispatcher")?;

    let cases = [
        ("perplexity", false, false),
        ("accuracy", true, false),
        ("accuracy", false, false),
        ("bleu", false, true),
        ("bleu", false, false),
        ("typo", false, false),
    ];
    for (m, l, r) in cases {
        println!("{m:>16}  labels={l} refs={r}  →  {:?}", dispatch(m, l, r));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perplexity_is_self_scoring() {
        assert_eq!(classify_metric("perplexity"), Some(MetricKind::SelfScoring));
        assert_eq!(dispatch("perplexity", false, false), DispatchVerdict::Ok);
    }

    #[test]
    fn accuracy_needs_labels() {
        assert_eq!(classify_metric("accuracy"), Some(MetricKind::NeedsLabels));
        assert_eq!(
            dispatch("accuracy", false, false),
            DispatchVerdict::MissingLabels
        );
        assert_eq!(dispatch("accuracy", true, false), DispatchVerdict::Ok);
    }

    #[test]
    fn bleu_needs_references() {
        assert_eq!(classify_metric("bleu"), Some(MetricKind::NeedsReferences));
        assert_eq!(
            dispatch("bleu", false, false),
            DispatchVerdict::MissingReferences
        );
        assert_eq!(dispatch("bleu", false, true), DispatchVerdict::Ok);
    }

    #[test]
    fn unknown_metric_rejected() {
        assert!(classify_metric("typo").is_none());
        assert_eq!(dispatch("typo", true, true), DispatchVerdict::UnknownMetric);
    }

    #[test]
    fn benchmark_suites_self_scoring() {
        // MMLU/HellaSwag/ARC/GSM8K bring their own labels in the dataset format.
        for m in ["mmlu", "hellaswag", "arc", "gsm8k"] {
            assert_eq!(classify_metric(m), Some(MetricKind::SelfScoring));
        }
    }

    #[test]
    fn rouge_l_and_exact_match_need_references() {
        for m in ["rouge_l", "exact_match"] {
            assert_eq!(classify_metric(m), Some(MetricKind::NeedsReferences));
        }
    }

    #[test]
    fn top_k_accuracy_needs_labels() {
        assert_eq!(
            classify_metric("top_k_accuracy"),
            Some(MetricKind::NeedsLabels)
        );
    }
}
