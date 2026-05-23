//! # apr eval — Dataset Dispatcher (wikitext-2 / lambada / custom)
//!
//! `apr eval --dataset <D>` chooses the eval corpus. Three known sources:
//! `wikitext-2` (default), `lambada`, `custom` (requires `--text`). This
//! recipe builds the dispatcher and asserts the contract: known datasets
//! resolve, `custom` without `--text` rejects, unknown datasets reject
//! rather than silently falling back to wikitext-2.
//!
//! Demonstrates the **EVAL.6** recipe for PMAT-103 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EVAL-001
//!
//! Run with: cargo run --example cli_eval_dataset_dispatcher
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSource {
    Wikitext2,
    Lambada,
    Custom,
}

#[derive(Debug, PartialEq)]
pub enum DatasetVerdict {
    Resolved(DatasetSource),
    UnknownDataset(String),
    CustomRequiresText,
    TextSpecifiedWithoutCustom,
}

pub fn dispatch(dataset: &str, text: Option<&str>) -> DatasetVerdict {
    let source = match dataset {
        "wikitext-2" => DatasetSource::Wikitext2,
        "lambada" => DatasetSource::Lambada,
        "custom" => DatasetSource::Custom,
        _ => return DatasetVerdict::UnknownDataset(dataset.into()),
    };
    match (source, text) {
        (DatasetSource::Custom, None) => DatasetVerdict::CustomRequiresText,
        (DatasetSource::Custom, Some(_)) => DatasetVerdict::Resolved(source),
        (_, Some(_)) => DatasetVerdict::TextSpecifiedWithoutCustom,
        (s, None) => DatasetVerdict::Resolved(s),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_dataset_dispatcher")?;

    let cases = [
        ("wikitext-2 default", "wikitext-2", None),
        ("lambada", "lambada", None),
        ("custom + text", "custom", Some("The cat sat on the mat.")),
        ("custom no text", "custom", None),
        ("text without custom", "wikitext-2", Some("anything")),
        ("typo dataset", "wikitex", None),
    ];
    for (label, dataset, text) in cases {
        println!("{label:>25}  →  {:?}", dispatch(dataset, text));
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
    fn wikitext_default_resolves() {
        assert_eq!(
            dispatch("wikitext-2", None),
            DatasetVerdict::Resolved(DatasetSource::Wikitext2)
        );
    }

    #[test]
    fn lambada_resolves() {
        assert_eq!(
            dispatch("lambada", None),
            DatasetVerdict::Resolved(DatasetSource::Lambada)
        );
    }

    #[test]
    fn custom_with_text_resolves() {
        assert_eq!(
            dispatch("custom", Some("hi")),
            DatasetVerdict::Resolved(DatasetSource::Custom)
        );
    }

    #[test]
    fn custom_without_text_rejected() {
        assert_eq!(dispatch("custom", None), DatasetVerdict::CustomRequiresText);
    }

    #[test]
    fn text_without_custom_rejected() {
        // Ambiguous — operator probably forgot `--dataset custom`.
        assert_eq!(
            dispatch("wikitext-2", Some("hi")),
            DatasetVerdict::TextSpecifiedWithoutCustom
        );
    }

    #[test]
    fn unknown_dataset_rejected() {
        assert!(matches!(
            dispatch("wikitex", None),
            DatasetVerdict::UnknownDataset(_)
        ));
    }

    #[test]
    fn empty_dataset_rejected_as_unknown() {
        assert!(matches!(
            dispatch("", None),
            DatasetVerdict::UnknownDataset(_)
        ));
    }
}
