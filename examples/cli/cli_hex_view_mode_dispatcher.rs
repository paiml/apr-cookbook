//! # apr hex — View Mode Dispatcher
//!
//! `apr hex <FILE>` accepts mutually-exclusive view modes:
//! `--header`, `--blocks`, `--distribution`, `--contract`, `--entropy`,
//! `--raw`. Without any flag the default is "tensor list with limited
//! preview". This recipe builds the dispatcher and asserts the contract:
//! at most one mode flag at a time, conflicts surface with a clear
//! diagnostic.
//!
//! Demonstrates the **HEX.6** recipe for PMAT-100 (apr hex coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HEX-003
//!
//! Run with: cargo run --example cli_hex_view_mode_dispatcher
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct HexFlags {
    pub header: bool,
    pub blocks: bool,
    pub distribution: bool,
    pub contract: bool,
    pub entropy: bool,
    pub raw: bool,
    pub list: bool,
    pub stats: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum HexViewMode {
    Default,
    Header,
    Blocks,
    Distribution,
    Contract,
    Entropy,
    Raw,
    List,
    Stats,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DispatchVerdict {
    Mode(HexViewMode),
    Conflict { selected: Vec<HexViewMode> },
}

pub fn dispatch(flags: HexFlags) -> DispatchVerdict {
    let mut chosen = Vec::new();
    if flags.header {
        chosen.push(HexViewMode::Header);
    }
    if flags.blocks {
        chosen.push(HexViewMode::Blocks);
    }
    if flags.distribution {
        chosen.push(HexViewMode::Distribution);
    }
    if flags.contract {
        chosen.push(HexViewMode::Contract);
    }
    if flags.entropy {
        chosen.push(HexViewMode::Entropy);
    }
    if flags.raw {
        chosen.push(HexViewMode::Raw);
    }
    if flags.list {
        chosen.push(HexViewMode::List);
    }
    if flags.stats {
        chosen.push(HexViewMode::Stats);
    }
    match chosen.len() {
        0 => DispatchVerdict::Mode(HexViewMode::Default),
        1 => DispatchVerdict::Mode(chosen.into_iter().next().unwrap()),
        _ => DispatchVerdict::Conflict { selected: chosen },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_hex_view_mode_dispatcher")?;

    let cases = [
        ("default", HexFlags::default()),
        (
            "header only",
            HexFlags {
                header: true,
                ..Default::default()
            },
        ),
        (
            "blocks only",
            HexFlags {
                blocks: true,
                ..Default::default()
            },
        ),
        (
            "conflict header+raw",
            HexFlags {
                header: true,
                raw: true,
                ..Default::default()
            },
        ),
        (
            "triple conflict",
            HexFlags {
                header: true,
                blocks: true,
                distribution: true,
                ..Default::default()
            },
        ),
    ];

    for (label, f) in cases {
        println!("{label:>20}  →  {:?}", dispatch(f));
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
    fn no_flags_is_default_mode() {
        assert_eq!(
            dispatch(HexFlags::default()),
            DispatchVerdict::Mode(HexViewMode::Default)
        );
    }

    #[test]
    fn single_flag_picks_that_mode() {
        assert_eq!(
            dispatch(HexFlags {
                header: true,
                ..Default::default()
            }),
            DispatchVerdict::Mode(HexViewMode::Header)
        );
        assert_eq!(
            dispatch(HexFlags {
                raw: true,
                ..Default::default()
            }),
            DispatchVerdict::Mode(HexViewMode::Raw)
        );
    }

    #[test]
    fn two_flags_yield_conflict() {
        let v = dispatch(HexFlags {
            header: true,
            blocks: true,
            ..Default::default()
        });
        if let DispatchVerdict::Conflict { selected } = v {
            assert_eq!(selected.len(), 2);
        } else {
            panic!("expected Conflict");
        }
    }

    #[test]
    fn many_flags_yield_conflict() {
        let v = dispatch(HexFlags {
            header: true,
            blocks: true,
            distribution: true,
            contract: true,
            entropy: true,
            raw: true,
            list: true,
            stats: true,
        });
        if let DispatchVerdict::Conflict { selected } = v {
            assert_eq!(selected.len(), 8);
        } else {
            panic!("expected Conflict");
        }
    }

    #[test]
    fn each_mode_independently_dispatchable() {
        // No mode is shadowed by another.
        let modes = [
            (
                HexFlags {
                    distribution: true,
                    ..Default::default()
                },
                HexViewMode::Distribution,
            ),
            (
                HexFlags {
                    contract: true,
                    ..Default::default()
                },
                HexViewMode::Contract,
            ),
            (
                HexFlags {
                    entropy: true,
                    ..Default::default()
                },
                HexViewMode::Entropy,
            ),
            (
                HexFlags {
                    list: true,
                    ..Default::default()
                },
                HexViewMode::List,
            ),
            (
                HexFlags {
                    stats: true,
                    ..Default::default()
                },
                HexViewMode::Stats,
            ),
        ];
        for (flags, expected) in modes {
            assert_eq!(dispatch(flags), DispatchVerdict::Mode(expected));
        }
    }
}
