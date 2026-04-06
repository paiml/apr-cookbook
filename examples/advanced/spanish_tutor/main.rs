#![allow(unused_imports)]
//! Demo P: Spanish Language Tutor - translation with grammar explanations.
//! QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971

use std::collections::HashMap;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo P: Spanish Language Tutor ===\n");
    let tutor = SpanishTutor::new();
    println!("Dictionary: {} entries\n", tutor.dictionary_size());
    for s in [
        "hola",
        "el libro",
        "la casa grande",
        "yo hablo español",
        "tengo hambre",
        "buenos días",
        "él es bueno",
    ] {
        println!("{}\n{}\n", "-".repeat(40), tutor.translate(s).format());
    }
    println!("=== Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_impls() {
        assert_eq!(format!("{}", PartOfSpeech::Noun), "noun");
        assert_eq!(format!("{}", Gender::Masculine), "masc");
        assert_eq!(format!("{}", Tense::Present), "present");
        assert_eq!(format!("{}", Person::First), "1st");
        assert_eq!(format!("{}", Number::Singular), "sing");
    }

    #[test]
    fn test_verb_conjugation() {
        let c = VerbConjugation::new("hablar", Tense::Present)
            .with_person_number(Person::First, Number::Singular);
        assert!(!c.is_irregular);
        let f = c.format();
        assert!(f.contains("hablar") && f.contains("present") && f.contains("1st"));
        assert!(
            VerbConjugation::new("ser", Tense::Present)
                .irregular()
                .is_irregular
        );
    }

    #[test]
    fn test_word_entry() {
        let e = WordEntry::new("casa", "house", PartOfSpeech::Noun)
            .with_alt("home")
            .with_gender(Gender::Feminine);
        assert_eq!(e.english.len(), 2);
        assert_eq!(e.gender, Some(Gender::Feminine));
        let f = e.format();
        assert!(f.contains("casa") && f.contains("house") && f.contains("noun"));
    }

    #[test]
    fn test_translation_result() {
        let r = TranslationResult::new("hola", "hello");
        assert!(!r.is_idiom);
        let ri = TranslationResult::new("tener hambre", "to be hungry").as_idiom("to have hunger");
        assert!(ri.is_idiom);
        assert_eq!(ri.literal_translation, Some("to have hunger".into()));
    }

    #[test]
    fn test_grammar_explanation() {
        let g = GrammarExplanation::new("Rule", "Expl").with_example("ej", "ex");
        assert_eq!(g.examples.len(), 1);
    }

    #[test]
    fn test_dictionary() {
        let d = SpanishDictionary::new();
        assert!(d.lookup("hola").is_some());
        assert!(d.lookup("Casa").is_some()); // case insensitive
        assert_eq!(d.lookup("casa").unwrap()[0].english[0], "house");
        for w in ["el", "la", "los", "las", "ser", "estar", "tener"] {
            assert!(d.lookup(w).is_some());
        }
        let soy = d.lookup("soy").unwrap();
        assert!(soy[0].conjugation.is_some());
        let (e, l) = d.check_idiom("tener hambre").unwrap();
        assert_eq!(e, "to be hungry");
        assert_eq!(l, "to have hunger");
    }

    #[test]
    fn test_tutor_translate() {
        let t = SpanishTutor::new();
        assert!(t.dictionary_size() > 0);
        let r1 = t.translate("hola");
        assert!(r1.english.contains("hello") || r1.english.contains("hi"));
        let r2 = t.translate("el libro");
        assert!(r2.word_breakdown.len() >= 2);
        assert!(t.translate("tener hambre").is_idiom);
        let r3 = t.translate("asdfqwerty");
        assert!(r3.english.contains("[asdfqwerty]"));
        assert!(r3
            .word_breakdown
            .iter()
            .any(|e| e.pos == PartOfSpeech::Unknown));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_translation_returns_result(word in "[a-z]{1,10}") {
            let r = SpanishTutor::new().translate(&word);
            prop_assert!(!r.spanish.is_empty() && !r.english.is_empty());
        }

        #[test]
        fn prop_dictionary_deterministic(word in "(hola|casa|libro|ser|estar)") {
            let d = SpanishDictionary::new();
            prop_assert_eq!(d.lookup(&word).is_some(), d.lookup(&word).is_some());
        }

        #[test]
        fn prop_word_entry_format(spanish in "[a-z]+", english in "[a-z]+") {
            let f = WordEntry::new(&spanish, &english, PartOfSpeech::Noun).format();
            prop_assert!(!f.is_empty() && f.contains(&spanish));
        }

        #[test]
        fn prop_conjugation_format(verb in "(hablar|comer|vivir)") {
            prop_assert!(VerbConjugation::new(&verb, Tense::Present).format().contains(&verb));
        }
    }
}
