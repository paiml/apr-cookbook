//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;
use std::collections::HashMap;

impl SpanishDictionary {
    pub fn populate(&mut self) {
        use Gender::{Feminine, Masculine};
        use Number::{Plural, Singular};
        use PartOfSpeech::{
            Adjective, Adverb, Article, Conjunction, Interjection, Noun, Preposition, Pronoun, Verb,
        };
        use Person::{First, Second, Third};
        // Articles
        for (w, g, n) in [
            ("el", Masculine, Singular),
            ("la", Feminine, Singular),
            ("los", Masculine, Plural),
            ("las", Feminine, Plural),
        ] {
            self.add(
                WordEntry::new(w, "the", Article)
                    .with_gender(g)
                    .with_number(n),
            );
        }
        self.add(
            WordEntry::new("un", "a/an", Article)
                .with_gender(Masculine)
                .with_number(Singular),
        );
        self.add(
            WordEntry::new("una", "a/an", Article)
                .with_gender(Feminine)
                .with_number(Singular),
        );
        // Pronouns
        for (w, e) in [("yo", "I"), ("él", "he"), ("ella", "she")] {
            self.add(WordEntry::new(w, e, Pronoun));
        }
        self.add(WordEntry::new("tú", "you", Pronoun).with_note("informal singular"));
        self.add(WordEntry::new("nosotros", "we", Pronoun).with_gender(Masculine));
        self.add(WordEntry::new("ellos", "they", Pronoun).with_gender(Masculine));
        self.add(WordEntry::new("ellas", "they", Pronoun).with_gender(Feminine));
        // Nouns
        for (w, e, g) in [
            ("casa", "house", Feminine),
            ("libro", "book", Masculine),
            ("perro", "dog", Masculine),
            ("gato", "cat", Masculine),
            ("tiempo", "time", Masculine),
            ("día", "day", Masculine),
            ("noche", "night", Feminine),
            ("hombre", "man", Masculine),
            ("mujer", "woman", Feminine),
            ("niño", "boy", Masculine),
            ("niña", "girl", Feminine),
        ] {
            self.add(WordEntry::new(w, e, Noun).with_gender(g));
        }
        self.add(
            WordEntry::new("agua", "water", Noun)
                .with_gender(Feminine)
                .with_note("Uses 'el' despite feminine"),
        );
        // Adjectives
        self.add(
            WordEntry::new("grande", "big", Adjective)
                .with_alt("large")
                .with_note("Same for masc/fem"),
        );
        for (w, e, g) in [
            ("pequeño", "small", Masculine),
            ("pequeña", "small", Feminine),
            ("bueno", "good", Masculine),
            ("buena", "good", Feminine),
            ("malo", "bad", Masculine),
            ("mala", "bad", Feminine),
            ("nuevo", "new", Masculine),
            ("viejo", "old", Masculine),
        ] {
            self.add(WordEntry::new(w, e, Adjective).with_gender(g));
        }
        // Verb infinitives
        let mk_inf = |w: &str, e: &str, irreg: bool| {
            let mut c = VerbConjugation::new(w, Tense::Infinitive);
            if irreg {
                c = c.irregular();
            }
            WordEntry::new(w, e, Verb).with_conjugation(c)
        };
        for (w, e, ir) in [
            ("ser", "to be", true),
            ("estar", "to be", true),
            ("tener", "to have", true),
            ("hacer", "to do", true),
            ("ir", "to go", true),
            ("comer", "to eat", false),
            ("hablar", "to speak", false),
            ("vivir", "to live", false),
        ] {
            self.add(mk_inf(w, e, ir));
        }
        // Conjugated verbs
        let mk_conj = |w: &str, e: &str, inf: &str, p: Person, n: Number, ir: bool| {
            let mut c = VerbConjugation::new(inf, Tense::Present).with_person_number(p, n);
            if ir {
                c = c.irregular();
            }
            WordEntry::new(w, e, Verb).with_conjugation(c)
        };
        for (w, e, inf, p, n) in [
            ("soy", "I am", "ser", First, Singular),
            ("eres", "you are", "ser", Second, Singular),
            ("es", "is", "ser", Third, Singular),
            ("somos", "we are", "ser", First, Plural),
            ("son", "they are", "ser", Third, Plural),
            ("estoy", "I am", "estar", First, Singular),
            ("está", "is", "estar", Third, Singular),
            ("están", "they are", "estar", Third, Plural),
            ("tengo", "I have", "tener", First, Singular),
            ("tiene", "has", "tener", Third, Singular),
            ("voy", "I go", "ir", First, Singular),
            ("va", "goes", "ir", Third, Singular),
        ] {
            self.add(mk_conj(w, e, inf, p, n, true));
        }
        for (w, e, inf, p, n) in [
            ("hablo", "I speak", "hablar", First, Singular),
            ("habla", "speaks", "hablar", Third, Singular),
            ("hablamos", "we speak", "hablar", First, Plural),
            ("como", "I eat", "comer", First, Singular),
            ("come", "eats", "comer", Third, Singular),
        ] {
            self.add(mk_conj(w, e, inf, p, n, false));
        }
        // Prepositions
        self.add(
            WordEntry::new("en", "in", Preposition)
                .with_alt("on")
                .with_alt("at"),
        );
        for (w, e) in [
            ("de", "of"),
            ("a", "to"),
            ("con", "with"),
            ("sin", "without"),
        ] {
            self.add(WordEntry::new(w, e, Preposition));
        }
        self.add(WordEntry::new("por", "for", Preposition).with_alt("by"));
        self.add(WordEntry::new("para", "for", Preposition).with_alt("in order to"));
        // Conjunctions
        for (w, e) in [
            ("y", "and"),
            ("o", "or"),
            ("pero", "but"),
            ("porque", "because"),
        ] {
            self.add(WordEntry::new(w, e, Conjunction));
        }
        self.add(WordEntry::new("que", "that", Conjunction).with_alt("which"));
        // Adverbs
        for (w, e) in [
            ("muy", "very"),
            ("bien", "well"),
            ("mal", "badly"),
            ("siempre", "always"),
            ("nunca", "never"),
            ("aquí", "here"),
            ("allí", "there"),
            ("ahora", "now"),
            ("hoy", "today"),
            ("mañana", "tomorrow"),
            ("ayer", "yesterday"),
        ] {
            self.add(WordEntry::new(w, e, Adverb));
        }
        self.add(WordEntry::new("mucho", "much", Adverb).with_alt("a lot"));
        self.add(WordEntry::new("poco", "little", Adverb).with_alt("few"));
        // Phrases
        for (w, e) in [
            ("buenos días", "good morning"),
            ("buenas noches", "good night"),
            ("por favor", "please"),
            ("adiós", "goodbye"),
        ] {
            self.add(WordEntry::new(w, e, Interjection));
        }
        self.add(WordEntry::new("gracias", "thank you", Interjection).with_alt("thanks"));
        self.add(WordEntry::new("hola", "hello", Interjection).with_alt("hi"));
        // Idioms
        for (es, (en, lit)) in [
            ("tener hambre", ("to be hungry", "to have hunger")),
            ("tener sed", ("to be thirsty", "to have thirst")),
            ("tener frío", ("to be cold", "to have cold")),
            ("tener calor", ("to be hot", "to have heat")),
            ("tener razón", ("to be right", "to have reason")),
            ("hacer falta", ("to be necessary", "to make lack")),
            ("dar igual", ("to not matter", "to give equal")),
            ("echar de menos", ("to miss (someone)", "to throw of less")),
        ] {
            self.idioms.insert(es.into(), (en.into(), lit.into()));
        }
    }
}

impl SpanishTutor {
    pub fn translate(&self, spanish: &str) -> TranslationResult {
        let norm = spanish.trim().to_lowercase();
        if let Some((english, literal)) = self.dictionary.check_idiom(&norm) {
            let mut r = TranslationResult::new(spanish, english).as_idiom(literal);
            r.add_grammar(GrammarExplanation::new(
                "Idiomatic Expression",
                "Meaning differs from literal translation.",
            ));
            return r;
        }
        if let Some(entries) = self.dictionary.lookup(&norm) {
            let e = &entries[0];
            let mut r = TranslationResult::new(spanish, &e.english.join(" / "));
            r.add_word(e.clone());
            self.annotate(&mut r, e);
            return r;
        }
        let words: Vec<&str> = norm.split_whitespace().collect();
        let mut eng = Vec::new();
        let mut r = TranslationResult::new(spanish, "");
        for w in &words {
            if let Some(entries) = self.dictionary.lookup(w) {
                let e = &entries[0];
                eng.push(e.english[0].clone());
                r.add_word(e.clone());
                self.annotate(&mut r, e);
            } else {
                eng.push(format!("[{w}]"));
                r.add_word(WordEntry::new(
                    w,
                    &format!("[unknown: {w}]"),
                    PartOfSpeech::Unknown,
                ));
            }
        }
        r.english = eng.join(" ");
        self.check_agreement(&mut r);
        r
    }

    pub fn annotate(&self, r: &mut TranslationResult, e: &WordEntry) {
        match e.pos {
            PartOfSpeech::Verb => {
                if let Some(ref c) = e.conjugation {
                    if c.is_irregular {
                        r.add_grammar(GrammarExplanation::new(
                            "Irregular Verb",
                            &format!("'{}' is irregular", c.infinitive),
                        ));
                    }
                    if matches!(c.tense, Tense::Present) {
                        r.add_grammar(
                            GrammarExplanation::new(
                                "Present Tense",
                                "Current actions, habits, general truths",
                            )
                            .with_example("Hablo español", "I speak Spanish"),
                        );
                    }
                }
            }
            PartOfSpeech::Article => {
                if let Some(g) = e.gender {
                    let gn = if matches!(g, Gender::Masculine) {
                        "masculine"
                    } else {
                        "feminine"
                    };
                    r.add_grammar(
                        GrammarExplanation::new(
                            "Article Agreement",
                            &format!("{} = {gn}", e.spanish),
                        )
                        .with_example("el libro", "the book (masc)")
                        .with_example("la casa", "the house (fem)"),
                    );
                }
            }
            PartOfSpeech::Adjective => {
                r.add_grammar(
                    GrammarExplanation::new(
                        "Adjective Placement",
                        "Most adjectives come AFTER the noun",
                    )
                    .with_example("la casa grande", "the big house"),
                );
                if e.gender.is_some() {
                    r.add_grammar(GrammarExplanation::new(
                        "Adjective Agreement",
                        "Must agree in gender and number with noun",
                    ));
                }
            }
            PartOfSpeech::Noun => {
                if let Some(g) = e.gender {
                    let gn = if matches!(g, Gender::Masculine) {
                        "masculine"
                    } else {
                        "feminine"
                    };
                    r.add_grammar(GrammarExplanation::new(
                        "Noun Gender",
                        &format!("'{}' is {gn}", e.spanish),
                    ));
                }
            }
            _ => {}
        }
    }

    pub fn check_agreement(&self, r: &mut TranslationResult) {
        let mut art_g: Option<Gender> = None;
        for e in &r.word_breakdown {
            if e.pos == PartOfSpeech::Article {
                art_g = e.gender;
            } else if matches!(e.pos, PartOfSpeech::Noun | PartOfSpeech::Adjective) {
                if let (Some(a), Some(w)) = (art_g, e.gender) {
                    if a != w && !matches!(a, Gender::Neuter) && !matches!(w, Gender::Neuter) {
                        r.add_grammar(GrammarExplanation::new(
                            "Agreement Check",
                            "Article and noun/adjective genders should match",
                        ));
                        return;
                    }
                }
            }
        }
    }
}
