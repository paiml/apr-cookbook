//! # TUI Keymap Register Chord
//!
//! Register key chord → action mappings; detect conflicts (same chord
//! mapped twice) and prefix conflicts (one chord is a prefix of
//! another). Returns sorted conflict list.
//!
//! Demonstrates the **TUI.149** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: emacs `define-key` shadowing rules; tmux/screen prefix-
//!  binding conflict semantics.
//!
//! Run with: cargo run --example tui_keymap_register_chord
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum KeymapVerdict {
    Ok {
        conflicts: Vec<String>,
        unique_chord_count: u32,
    },
    InvalidConfig,
}

pub fn register(bindings: &[(&str, &str)]) -> KeymapVerdict {
    if bindings.is_empty() {
        return KeymapVerdict::InvalidConfig;
    }
    let mut chords: BTreeSet<String> = BTreeSet::new();
    let mut conflict_set: BTreeSet<String> = BTreeSet::new();
    // Detect duplicate-chord and prefix conflicts.
    for (chord, _) in bindings {
        if !chords.insert((*chord).to_string()) {
            conflict_set.insert((*chord).to_string());
        }
    }
    // Prefix conflict: if chord A is a strict prefix of chord B.
    let chords_vec: Vec<String> = chords.iter().cloned().collect();
    for i in 0..chords_vec.len() {
        for j in 0..chords_vec.len() {
            if i == j {
                continue;
            }
            if chords_vec[j].starts_with(&chords_vec[i])
                && chords_vec[j].len() > chords_vec[i].len()
                && chords_vec[j].as_bytes().get(chords_vec[i].len()).copied() == Some(b' ')
            {
                conflict_set.insert(chords_vec[i].clone());
                conflict_set.insert(chords_vec[j].clone());
            }
        }
    }
    KeymapVerdict::Ok {
        conflicts: conflict_set.into_iter().collect(),
        unique_chord_count: chords.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_keymap_register_chord")?;

    let b = [("Ctrl+S", "save"), ("Ctrl+S", "save-as")];
    println!("dup: {:?}", register(&b));
    let b2 = [("C-x", "menu"), ("C-x C-f", "open")];
    println!("prefix: {:?}", register(&b2));
    println!("invalid: {:?}", register(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registrar_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_conflict_clean() {
        let v = register(&[("Ctrl+S", "save"), ("Ctrl+O", "open")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert!(conflicts.is_empty());
        }
    }

    #[test]
    fn duplicate_chord_flagged() {
        let v = register(&[("Ctrl+S", "save"), ("Ctrl+S", "save-as")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["Ctrl+S".to_string()]);
        }
    }

    #[test]
    fn prefix_conflict_flagged() {
        let v = register(&[("C-x", "menu"), ("C-x C-f", "open")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert!(conflicts.contains(&"C-x".to_string()));
            assert!(conflicts.contains(&"C-x C-f".to_string()));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(register(&[]), KeymapVerdict::InvalidConfig);
    }

    #[test]
    fn unique_count_correct() {
        let v = register(&[("a", "x"), ("b", "y"), ("c", "z")]);
        if let KeymapVerdict::Ok {
            unique_chord_count, ..
        } = v
        {
            assert_eq!(unique_chord_count, 3);
        }
    }

    #[test]
    fn conflicts_sorted() {
        let v = register(&[("zeta", "x"), ("zeta", "y")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = register(&[("a", "x")]);
        let r2 = register(&[("a", "x")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_prefix_no_conflict() {
        // "abc" is not a chord-prefix of "abcdef" — no space-boundary.
        let v = register(&[("abc", "x"), ("abcdef", "y")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert!(conflicts.is_empty());
        }
    }

    #[test]
    fn unicode_chord_supported() {
        let v = register(&[("⌘+S", "save")]);
        if let KeymapVerdict::Ok {
            unique_chord_count, ..
        } = v
        {
            assert_eq!(unique_chord_count, 1);
        }
    }

    #[test]
    fn many_bindings_handled() {
        let b: Vec<(&str, &str)> = (0..30).map(|_| ("k", "v")).collect();
        let v = register(&b);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["k".to_string()]);
        }
    }

    #[test]
    fn three_chord_prefix_chain() {
        let v = register(&[("a", "x"), ("a b", "y"), ("a b c", "z")]);
        if let KeymapVerdict::Ok { conflicts, .. } = v {
            // a < a b, a b < a b c → all three conflict
            assert_eq!(conflicts.len(), 3);
        }
    }
}
