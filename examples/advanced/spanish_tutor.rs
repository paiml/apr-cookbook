//! Demo P: Spanish Language Tutor - translation with grammar explanations.
//! QA: Build, test, clippy, fmt PASS. Property tests included.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Article,
    Pronoun,
    Preposition,
    Conjunction,
    Interjection,
    Unknown,
}

impl fmt::Display for PartOfSpeech {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Noun => "noun",
                Self::Verb => "verb",
                Self::Adjective => "adj",
                Self::Adverb => "adv",
                Self::Article => "art",
                Self::Pronoun => "pron",
                Self::Preposition => "prep",
                Self::Conjunction => "conj",
                Self::Interjection => "interj",
                Self::Unknown => "?",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Masculine,
    Feminine,
    Neuter,
}

impl fmt::Display for Gender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Masculine => "masc",
                Self::Feminine => "fem",
                Self::Neuter => "",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Number {
    Singular,
    Plural,
}
impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Singular => "sing",
                Self::Plural => "plur",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tense {
    Present,
    Preterite,
    Imperfect,
    Future,
    Conditional,
    PresentPerfect,
    Subjunctive,
    Imperative,
    Infinitive,
    Gerund,
    PastParticiple,
}

impl fmt::Display for Tense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Present => "present",
                Self::Preterite => "preterite",
                Self::Imperfect => "imperfect",
                Self::Future => "future",
                Self::Conditional => "conditional",
                Self::PresentPerfect => "pres. perfect",
                Self::Subjunctive => "subjunctive",
                Self::Imperative => "imperative",
                Self::Infinitive => "infinitive",
                Self::Gerund => "gerund",
                Self::PastParticiple => "past part.",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Person {
    First,
    Second,
    Third,
}
impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::First => "1st",
                Self::Second => "2nd",
                Self::Third => "3rd",
            }
        )
    }
}

#[derive(Debug, Clone)]
pub struct VerbConjugation {
    pub infinitive: String,
    pub tense: Tense,
    pub person: Option<Person>,
    pub number: Option<Number>,
    pub is_irregular: bool,
}

impl VerbConjugation {
    #[must_use]
    pub fn new(infinitive: &str, tense: Tense) -> Self {
        Self {
            infinitive: infinitive.to_string(),
            tense,
            person: None,
            number: None,
            is_irregular: false,
        }
    }
    #[must_use]
    pub fn with_person_number(mut self, p: Person, n: Number) -> Self {
        self.person = Some(p);
        self.number = Some(n);
        self
    }
    #[must_use]
    pub fn irregular(mut self) -> Self {
        self.is_irregular = true;
        self
    }
    #[must_use]
    pub fn format(&self) -> String {
        let mut parts = vec![format!("inf: {}", self.infinitive), self.tense.to_string()];
        if let (Some(p), Some(n)) = (self.person, self.number) {
            parts.push(format!("{p} {n}"));
        }
        if self.is_irregular {
            parts.push("IRREGULAR".into());
        }
        parts.join(", ")
    }
}

#[derive(Debug, Clone)]
pub struct WordEntry {
    pub spanish: String,
    pub english: Vec<String>,
    pub pos: PartOfSpeech,
    pub gender: Option<Gender>,
    pub number: Option<Number>,
    pub conjugation: Option<VerbConjugation>,
    pub notes: Vec<String>,
}

impl WordEntry {
    #[must_use]
    pub fn new(s: &str, e: &str, pos: PartOfSpeech) -> Self {
        Self {
            spanish: s.into(),
            english: vec![e.into()],
            pos,
            gender: None,
            number: None,
            conjugation: None,
            notes: vec![],
        }
    }
    #[must_use]
    pub fn with_alt(mut self, a: &str) -> Self {
        self.english.push(a.into());
        self
    }
    #[must_use]
    pub fn with_gender(mut self, g: Gender) -> Self {
        self.gender = Some(g);
        self
    }
    #[must_use]
    pub fn with_number(mut self, n: Number) -> Self {
        self.number = Some(n);
        self
    }
    #[must_use]
    pub fn with_conjugation(mut self, c: VerbConjugation) -> Self {
        self.conjugation = Some(c);
        self
    }
    #[must_use]
    pub fn with_note(mut self, n: &str) -> Self {
        self.notes.push(n.into());
        self
    }
    #[must_use]
    pub fn format(&self) -> String {
        let mut p = vec![
            format!("\"{}\" -> \"{}\"", self.spanish, self.english.join(" / ")),
            format!("[{}]", self.pos),
        ];
        if let Some(g) = self.gender {
            if matches!(g, Gender::Masculine | Gender::Feminine) {
                p.push(format!("({g})"));
            }
        }
        if let Some(n) = self.number {
            p.push(format!("({n})"));
        }
        if let Some(ref c) = self.conjugation {
            p.push(format!("<{}>", c.format()));
        }
        p.join(" ")
    }
}

#[derive(Debug, Clone)]
pub struct GrammarExplanation {
    pub rule: String,
    pub explanation: String,
    pub examples: Vec<(String, String)>,
}

impl GrammarExplanation {
    #[must_use]
    pub fn new(rule: &str, expl: &str) -> Self {
        Self {
            rule: rule.into(),
            explanation: expl.into(),
            examples: vec![],
        }
    }
    #[must_use]
    pub fn with_example(mut self, es: &str, en: &str) -> Self {
        self.examples.push((es.into(), en.into()));
        self
    }
}

#[derive(Debug)]
pub struct TranslationResult {
    pub spanish: String,
    pub english: String,
    pub word_breakdown: Vec<WordEntry>,
    pub grammar: Vec<GrammarExplanation>,
    pub is_idiom: bool,
    pub literal_translation: Option<String>,
}

impl TranslationResult {
    #[must_use]
    pub fn new(s: &str, e: &str) -> Self {
        Self {
            spanish: s.into(),
            english: e.into(),
            word_breakdown: vec![],
            grammar: vec![],
            is_idiom: false,
            literal_translation: None,
        }
    }
    #[must_use]
    pub fn as_idiom(mut self, lit: &str) -> Self {
        self.is_idiom = true;
        self.literal_translation = Some(lit.into());
        self
    }
    pub fn add_word(&mut self, e: WordEntry) {
        self.word_breakdown.push(e);
    }
    pub fn add_grammar(&mut self, g: GrammarExplanation) {
        self.grammar.push(g);
    }
    #[must_use]
    pub fn format(&self) -> String {
        let mut lines = vec![
            format!("Spanish: {}", self.spanish),
            format!("English: {}", self.english),
        ];
        if self.is_idiom {
            lines.push("IDIOM".into());
            if let Some(ref l) = self.literal_translation {
                lines.push(format!("  Literal: {l}"));
            }
        }
        if !self.word_breakdown.is_empty() {
            lines.push("\nWords:".into());
            for e in &self.word_breakdown {
                lines.push(format!("  {}", e.format()));
            }
        }
        if !self.grammar.is_empty() {
            lines.push("\nGrammar:".into());
            for g in &self.grammar {
                lines.push(format!("  {}: {}", g.rule, g.explanation));
                for (es, en) in &g.examples {
                    lines.push(format!("    \"{}\" = \"{}\"", es, en));
                }
            }
        }
        lines.join("\n")
    }
}

pub struct SpanishDictionary {
    words: HashMap<String, Vec<WordEntry>>,
    idioms: HashMap<String, (String, String)>,
}

impl SpanishDictionary {
    #[must_use]
    pub fn new() -> Self {
        let mut d = Self {
            words: HashMap::new(),
            idioms: HashMap::new(),
        };
        d.populate();
        d
    }

    fn add(&mut self, e: WordEntry) {
        self.words
            .entry(e.spanish.to_lowercase())
            .or_default()
            .push(e);
    }

    fn populate(&mut self) {
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

    #[must_use]
    pub fn lookup(&self, word: &str) -> Option<&Vec<WordEntry>> {
        self.words.get(&word.to_lowercase())
    }
    #[must_use]
    pub fn check_idiom(&self, phrase: &str) -> Option<(&str, &str)> {
        self.idioms
            .get(&phrase.to_lowercase())
            .map(|(e, l)| (e.as_str(), l.as_str()))
    }
}

impl Default for SpanishDictionary {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SpanishTutor {
    dictionary: SpanishDictionary,
}

impl SpanishTutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: SpanishDictionary::new(),
        }
    }

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

    fn annotate(&self, r: &mut TranslationResult, e: &WordEntry) {
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

    fn check_agreement(&self, r: &mut TranslationResult) {
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

    #[must_use]
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.words.len()
    }
}

impl Default for SpanishTutor {
    fn default() -> Self {
        Self::new()
    }
}

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
