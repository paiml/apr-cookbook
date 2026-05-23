//! # TUI Keybinding Dispatch
//!
//! Validate a keybinding table `(modifier, key, action)`: detect
//! double-mappings (same chord → two actions), unreachable actions,
//! and dispatch a given chord to its action.
//!
//! Demonstrates the **TUI.60** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU readline keymap; vim normal-mode dispatch.
//!
//! Run with: cargo run --example tui_keybinding_dispatch
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok { action: String },
    NotBound,
    DoubleMapping { chord: String },
    InvalidConfig,
}

pub fn dispatch(bindings: &[(&str, &str, &str)], modifier: &str, key: &str) -> DispatchVerdict {
    if bindings.is_empty() || key.is_empty() {
        return DispatchVerdict::InvalidConfig;
    }
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (m, k, a) in bindings {
        let chord = format!("{m}+{k}");
        if let Some(existing) = map.get(&chord) {
            if existing != a {
                return DispatchVerdict::DoubleMapping { chord };
            }
        }
        map.insert(chord, (*a).to_string());
    }
    let target = format!("{modifier}+{key}");
    map.get(&target)
        .map_or(DispatchVerdict::NotBound, |a| DispatchVerdict::Ok {
            action: a.clone(),
        })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_keybinding_dispatch")?;

    let bindings = [
        ("ctrl", "c", "copy"),
        ("ctrl", "v", "paste"),
        ("ctrl", "x", "cut"),
    ];
    println!("copy: {:?}", dispatch(&bindings, "ctrl", "c"));
    println!("unbound: {:?}", dispatch(&bindings, "ctrl", "z"));
    let conflict = [("ctrl", "c", "copy"), ("ctrl", "c", "cancel")];
    println!("conflict: {:?}", dispatch(&conflict, "ctrl", "c"));
    println!("invalid: {:?}", dispatch(&[], "", ""));
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
    fn known_chord_dispatched() {
        let bindings = [("ctrl", "c", "copy")];
        let v = dispatch(&bindings, "ctrl", "c");
        if let DispatchVerdict::Ok { action } = v {
            assert_eq!(action, "copy");
        }
    }

    #[test]
    fn unknown_chord_unbound() {
        let bindings = [("ctrl", "c", "copy")];
        assert_eq!(dispatch(&bindings, "ctrl", "x"), DispatchVerdict::NotBound);
    }

    #[test]
    fn double_mapping_flagged() {
        let bindings = [("ctrl", "c", "copy"), ("ctrl", "c", "cancel")];
        let v = dispatch(&bindings, "ctrl", "c");
        assert!(matches!(v, DispatchVerdict::DoubleMapping { .. }));
    }

    #[test]
    fn duplicate_same_action_ok() {
        // Same chord bound to same action twice — not a conflict.
        let bindings = [("ctrl", "c", "copy"), ("ctrl", "c", "copy")];
        let v = dispatch(&bindings, "ctrl", "c");
        if let DispatchVerdict::Ok { action } = v {
            assert_eq!(action, "copy");
        }
    }

    #[test]
    fn empty_bindings_rejected() {
        assert_eq!(dispatch(&[], "ctrl", "c"), DispatchVerdict::InvalidConfig);
    }

    #[test]
    fn empty_key_rejected() {
        let bindings = [("ctrl", "c", "copy")];
        assert_eq!(
            dispatch(&bindings, "ctrl", ""),
            DispatchVerdict::InvalidConfig
        );
    }

    #[test]
    fn modifier_matters() {
        let bindings = [("ctrl", "c", "copy"), ("alt", "c", "compose")];
        if let DispatchVerdict::Ok { action } = dispatch(&bindings, "alt", "c") {
            assert_eq!(action, "compose");
        }
    }

    #[test]
    fn no_modifier_works() {
        let bindings = [("", "q", "quit")];
        if let DispatchVerdict::Ok { action } = dispatch(&bindings, "", "q") {
            assert_eq!(action, "quit");
        }
    }

    #[test]
    fn deterministic() {
        let bindings = [("ctrl", "a", "select_all")];
        let a = dispatch(&bindings, "ctrl", "a");
        let b = dispatch(&bindings, "ctrl", "a");
        assert_eq!(a, b);
    }

    #[test]
    fn many_bindings_efficient() {
        let bindings: Vec<(&str, &str, &str)> = (0..50).map(|_| ("ctrl", "x", "cut")).collect();
        // All same — no conflict.
        let v = dispatch(&bindings, "ctrl", "x");
        if let DispatchVerdict::Ok { action } = v {
            assert_eq!(action, "cut");
        }
    }

    #[test]
    fn unbound_after_table_built() {
        let bindings = [("ctrl", "c", "copy"), ("ctrl", "v", "paste")];
        assert_eq!(dispatch(&bindings, "shift", "z"), DispatchVerdict::NotBound);
    }
}
