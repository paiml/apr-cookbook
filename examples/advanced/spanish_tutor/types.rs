#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use proptest::prelude::*;
#[allow(unused_imports)]
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
    #[allow(clippy::wrong_self_convention)]
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
    pub words: HashMap<String, Vec<WordEntry>>,
    pub idioms: HashMap<String, (String, String)>,
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

    pub fn add(&mut self, e: WordEntry) {
        self.words
            .entry(e.spanish.to_lowercase())
            .or_default()
            .push(e);
    }

    // populate() moved to helpers.rs

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
    pub dictionary: SpanishDictionary,
}

impl SpanishTutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: SpanishDictionary::new(),
        }
    }

    // translate(), annotate(), check_agreement() moved to helpers.rs

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
