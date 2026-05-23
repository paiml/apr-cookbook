//! # apr inspect — View Mode Dispatcher
//!
//! `apr inspect <FILE>` accepts mutually-exclusive view flags:
//! `--vocab`, `--filters`, `--weights`. Without any flag the default
//! is the metadata summary. This recipe builds the dispatcher and
//! asserts: at most one mode flag at a time, conflicts surface clearly.
//!
//! Demonstrates the **INSPECT.6** recipe for PMAT-109 (apr inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender INSPECT-001
//!
//! Run with: cargo run --example cli_inspect_view_mode_dispatcher
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct InspectFlags {
    pub vocab: bool,
    pub filters: bool,
    pub weights: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum InspectMode {
    Summary,
    Vocab,
    Filters,
    Weights,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DispatchVerdict {
    Mode(InspectMode),
    Conflict { selected: Vec<InspectMode> },
}

pub fn dispatch(flags: InspectFlags) -> DispatchVerdict {
    let mut chosen = Vec::new();
    if flags.vocab {
        chosen.push(InspectMode::Vocab);
    }
    if flags.filters {
        chosen.push(InspectMode::Filters);
    }
    if flags.weights {
        chosen.push(InspectMode::Weights);
    }
    match chosen.len() {
        0 => DispatchVerdict::Mode(InspectMode::Summary),
        1 => DispatchVerdict::Mode(chosen.into_iter().next().unwrap()),
        _ => DispatchVerdict::Conflict { selected: chosen },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_inspect_view_mode_dispatcher")?;

    let cases = [
        ("default", InspectFlags::default()),
        (
            "vocab",
            InspectFlags {
                vocab: true,
                ..Default::default()
            },
        ),
        (
            "filters",
            InspectFlags {
                filters: true,
                ..Default::default()
            },
        ),
        (
            "weights",
            InspectFlags {
                weights: true,
                ..Default::default()
            },
        ),
        (
            "vocab+weights",
            InspectFlags {
                vocab: true,
                weights: true,
                ..Default::default()
            },
        ),
        (
            "all three",
            InspectFlags {
                vocab: true,
                filters: true,
                weights: true,
            },
        ),
    ];
    for (label, f) in cases {
        println!("{label:>16}  →  {:?}", dispatch(f));
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
    fn no_flags_yields_summary() {
        assert_eq!(
            dispatch(InspectFlags::default()),
            DispatchVerdict::Mode(InspectMode::Summary)
        );
    }

    #[test]
    fn single_vocab_flag_picks_vocab() {
        assert_eq!(
            dispatch(InspectFlags {
                vocab: true,
                ..Default::default()
            }),
            DispatchVerdict::Mode(InspectMode::Vocab)
        );
    }

    #[test]
    fn single_filters_flag_picks_filters() {
        assert_eq!(
            dispatch(InspectFlags {
                filters: true,
                ..Default::default()
            }),
            DispatchVerdict::Mode(InspectMode::Filters)
        );
    }

    #[test]
    fn single_weights_flag_picks_weights() {
        assert_eq!(
            dispatch(InspectFlags {
                weights: true,
                ..Default::default()
            }),
            DispatchVerdict::Mode(InspectMode::Weights)
        );
    }

    #[test]
    fn two_flags_yield_conflict() {
        let v = dispatch(InspectFlags {
            vocab: true,
            weights: true,
            ..Default::default()
        });
        if let DispatchVerdict::Conflict { selected } = v {
            assert_eq!(selected.len(), 2);
        } else {
            panic!("expected Conflict");
        }
    }

    #[test]
    fn three_flags_yield_conflict() {
        let v = dispatch(InspectFlags {
            vocab: true,
            filters: true,
            weights: true,
        });
        if let DispatchVerdict::Conflict { selected } = v {
            assert_eq!(selected.len(), 3);
        }
    }
}
