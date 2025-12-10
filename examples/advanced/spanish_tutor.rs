//! # Demo P: Spanish Language Tutor
//!
//! Translates Spanish words/phrases to English with grammar explanations.
//! Highlights grammatical structures including verb conjugations, gender,
//! number agreement, and sentence structure.
//!
//! ## Toyota Way Principles
//!
//! - **Poka-yoke**: Clear grammar rules prevent learning errors
//! - **Genchi Genbutsu**: Learn from actual language patterns
//! - **Kaizen**: Progressive difficulty levels
//!
//! ## Features
//!
//! - Word-by-word translation with part-of-speech tagging
//! - Verb conjugation analysis (tense, mood, person)
//! - Gender and number agreement checking
//! - Idiomatic expression recognition

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Grammar Types
// ============================================================================

/// Part of speech
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
        let s = match self {
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
        };
        write!(f, "{s}")
    }
}

/// Grammatical gender
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Masculine,
    Feminine,
    Neuter, // For words without gender
}

impl fmt::Display for Gender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Masculine => "masc",
            Self::Feminine => "fem",
            Self::Neuter => "",
        };
        write!(f, "{s}")
    }
}

/// Grammatical number
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Number {
    Singular,
    Plural,
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Singular => "sing",
            Self::Plural => "plur",
        };
        write!(f, "{s}")
    }
}

/// Verb tense
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tense {
    Present,
    Preterite, // Simple past
    Imperfect, // Past habitual/ongoing
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
        let s = match self {
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
        };
        write!(f, "{s}")
    }
}

/// Person (1st, 2nd, 3rd)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Person {
    First,
    Second,
    Third,
}

impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::First => "1st",
            Self::Second => "2nd",
            Self::Third => "3rd",
        };
        write!(f, "{s}")
    }
}

/// Verb conjugation info
#[derive(Debug, Clone)]
pub struct VerbConjugation {
    pub infinitive: String,
    pub tense: Tense,
    pub person: Option<Person>,
    pub number: Option<Number>,
    pub is_irregular: bool,
}

impl VerbConjugation {
    /// Create new conjugation
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

    /// Set person and number
    #[must_use]
    pub fn with_person_number(mut self, person: Person, number: Number) -> Self {
        self.person = Some(person);
        self.number = Some(number);
        self
    }

    /// Mark as irregular
    #[must_use]
    pub fn irregular(mut self) -> Self {
        self.is_irregular = true;
        self
    }

    /// Format conjugation info
    #[must_use]
    pub fn format(&self) -> String {
        let mut parts = vec![format!("inf: {}", self.infinitive), self.tense.to_string()];
        if let (Some(p), Some(n)) = (self.person, self.number) {
            parts.push(format!("{p} {n}"));
        }
        if self.is_irregular {
            parts.push("IRREGULAR".to_string());
        }
        parts.join(", ")
    }
}

// ============================================================================
// Word Entry
// ============================================================================

/// Dictionary entry for a word
#[derive(Debug, Clone)]
pub struct WordEntry {
    /// Spanish word
    pub spanish: String,
    /// English translation(s)
    pub english: Vec<String>,
    /// Part of speech
    pub pos: PartOfSpeech,
    /// Gender (for nouns/adjectives)
    pub gender: Option<Gender>,
    /// Number
    pub number: Option<Number>,
    /// Verb conjugation (if verb)
    pub conjugation: Option<VerbConjugation>,
    /// Usage notes
    pub notes: Vec<String>,
}

impl WordEntry {
    /// Create new entry
    #[must_use]
    pub fn new(spanish: &str, english: &str, pos: PartOfSpeech) -> Self {
        Self {
            spanish: spanish.to_string(),
            english: vec![english.to_string()],
            pos,
            gender: None,
            number: None,
            conjugation: None,
            notes: Vec::new(),
        }
    }

    /// Add alternative translation
    #[must_use]
    pub fn with_alt(mut self, alt: &str) -> Self {
        self.english.push(alt.to_string());
        self
    }

    /// Set gender
    #[must_use]
    pub fn with_gender(mut self, gender: Gender) -> Self {
        self.gender = Some(gender);
        self
    }

    /// Set number
    #[must_use]
    pub fn with_number(mut self, number: Number) -> Self {
        self.number = Some(number);
        self
    }

    /// Set conjugation
    #[must_use]
    pub fn with_conjugation(mut self, conj: VerbConjugation) -> Self {
        self.conjugation = Some(conj);
        self
    }

    /// Add note
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        self.notes.push(note.to_string());
        self
    }

    /// Format entry for display
    #[must_use]
    pub fn format(&self) -> String {
        let mut parts = vec![
            format!("\"{}\" → \"{}\"", self.spanish, self.english.join(" / ")),
            format!("[{}]", self.pos),
        ];

        if let Some(g) = self.gender {
            if matches!(g, Gender::Masculine | Gender::Feminine) {
                parts.push(format!("({g})"));
            }
        }

        if let Some(n) = self.number {
            parts.push(format!("({n})"));
        }

        if let Some(ref conj) = self.conjugation {
            parts.push(format!("⟨{}⟩", conj.format()));
        }

        parts.join(" ")
    }
}

// ============================================================================
// Grammar Rules
// ============================================================================

/// Grammar explanation
#[derive(Debug, Clone)]
pub struct GrammarExplanation {
    /// Rule name
    pub rule: String,
    /// Explanation
    pub explanation: String,
    /// Examples
    pub examples: Vec<(String, String)>,
}

impl GrammarExplanation {
    /// Create new explanation
    #[must_use]
    pub fn new(rule: &str, explanation: &str) -> Self {
        Self {
            rule: rule.to_string(),
            explanation: explanation.to_string(),
            examples: Vec::new(),
        }
    }

    /// Add example
    #[must_use]
    pub fn with_example(mut self, spanish: &str, english: &str) -> Self {
        self.examples
            .push((spanish.to_string(), english.to_string()));
        self
    }
}

// ============================================================================
// Translation Result
// ============================================================================

/// Result of translating a phrase
#[derive(Debug)]
pub struct TranslationResult {
    /// Original Spanish text
    pub spanish: String,
    /// English translation
    pub english: String,
    /// Word-by-word breakdown
    pub word_breakdown: Vec<WordEntry>,
    /// Grammar explanations
    pub grammar: Vec<GrammarExplanation>,
    /// Is this an idiom?
    pub is_idiom: bool,
    /// Literal translation (for idioms)
    pub literal_translation: Option<String>,
}

impl TranslationResult {
    /// Create new result
    #[must_use]
    pub fn new(spanish: &str, english: &str) -> Self {
        Self {
            spanish: spanish.to_string(),
            english: english.to_string(),
            word_breakdown: Vec::new(),
            grammar: Vec::new(),
            is_idiom: false,
            literal_translation: None,
        }
    }

    /// Mark as idiom
    #[must_use]
    pub fn as_idiom(mut self, literal: &str) -> Self {
        self.is_idiom = true;
        self.literal_translation = Some(literal.to_string());
        self
    }

    /// Add word breakdown
    pub fn add_word(&mut self, entry: WordEntry) {
        self.word_breakdown.push(entry);
    }

    /// Add grammar explanation
    pub fn add_grammar(&mut self, explanation: GrammarExplanation) {
        self.grammar.push(explanation);
    }

    /// Format complete analysis
    #[must_use]
    pub fn format(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("📝 Spanish: {}", self.spanish));
        lines.push(format!("🔤 English: {}", self.english));

        if self.is_idiom {
            lines.push("⚡ This is an IDIOM".to_string());
            if let Some(ref lit) = self.literal_translation {
                lines.push(format!("   Literal: {lit}"));
            }
        }

        if !self.word_breakdown.is_empty() {
            lines.push("\n📖 Word Breakdown:".to_string());
            for entry in &self.word_breakdown {
                lines.push(format!("   {}", entry.format()));
            }
        }

        if !self.grammar.is_empty() {
            lines.push("\n📚 Grammar Notes:".to_string());
            for g in &self.grammar {
                lines.push(format!("   ▸ {}: {}", g.rule, g.explanation));
                for (es, en) in &g.examples {
                    lines.push(format!("     Ex: \"{}\" = \"{}\"", es, en));
                }
            }
        }

        lines.join("\n")
    }
}

// ============================================================================
// Dictionary
// ============================================================================

/// Spanish-English dictionary with grammar support
pub struct SpanishDictionary {
    words: HashMap<String, Vec<WordEntry>>,
    idioms: HashMap<String, (String, String)>, // (english, literal)
    #[allow(dead_code)] // Reserved for future conjugation features
    verb_conjugations: HashMap<String, VerbConjugation>,
}

impl SpanishDictionary {
    /// Create new dictionary with common words
    #[must_use]
    pub fn new() -> Self {
        let mut dict = Self {
            words: HashMap::new(),
            idioms: HashMap::new(),
            verb_conjugations: HashMap::new(),
        };
        dict.populate_common_words();
        dict.populate_idioms();
        dict.populate_verb_conjugations();
        dict
    }

    fn populate_common_words(&mut self) {
        // Articles
        self.add_word(
            WordEntry::new("el", "the", PartOfSpeech::Article)
                .with_gender(Gender::Masculine)
                .with_number(Number::Singular),
        );
        self.add_word(
            WordEntry::new("la", "the", PartOfSpeech::Article)
                .with_gender(Gender::Feminine)
                .with_number(Number::Singular),
        );
        self.add_word(
            WordEntry::new("los", "the", PartOfSpeech::Article)
                .with_gender(Gender::Masculine)
                .with_number(Number::Plural),
        );
        self.add_word(
            WordEntry::new("las", "the", PartOfSpeech::Article)
                .with_gender(Gender::Feminine)
                .with_number(Number::Plural),
        );
        self.add_word(
            WordEntry::new("un", "a/an", PartOfSpeech::Article)
                .with_gender(Gender::Masculine)
                .with_number(Number::Singular),
        );
        self.add_word(
            WordEntry::new("una", "a/an", PartOfSpeech::Article)
                .with_gender(Gender::Feminine)
                .with_number(Number::Singular),
        );

        // Pronouns
        self.add_word(WordEntry::new("yo", "I", PartOfSpeech::Pronoun));
        self.add_word(
            WordEntry::new("tú", "you", PartOfSpeech::Pronoun).with_note("informal singular"),
        );
        self.add_word(WordEntry::new("él", "he", PartOfSpeech::Pronoun));
        self.add_word(WordEntry::new("ella", "she", PartOfSpeech::Pronoun));
        self.add_word(
            WordEntry::new("nosotros", "we", PartOfSpeech::Pronoun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("nosotras", "we", PartOfSpeech::Pronoun).with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("ellos", "they", PartOfSpeech::Pronoun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("ellas", "they", PartOfSpeech::Pronoun).with_gender(Gender::Feminine),
        );

        // Common nouns
        self.add_word(
            WordEntry::new("casa", "house", PartOfSpeech::Noun)
                .with_alt("home")
                .with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("libro", "book", PartOfSpeech::Noun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("perro", "dog", PartOfSpeech::Noun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("gato", "cat", PartOfSpeech::Noun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("agua", "water", PartOfSpeech::Noun)
                .with_gender(Gender::Feminine)
                .with_note("Uses 'el' despite being feminine: 'el agua'"),
        );
        self.add_word(
            WordEntry::new("tiempo", "time", PartOfSpeech::Noun)
                .with_alt("weather")
                .with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("día", "day", PartOfSpeech::Noun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("noche", "night", PartOfSpeech::Noun).with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("hombre", "man", PartOfSpeech::Noun).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("mujer", "woman", PartOfSpeech::Noun).with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("niño", "boy", PartOfSpeech::Noun)
                .with_alt("child")
                .with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("niña", "girl", PartOfSpeech::Noun).with_gender(Gender::Feminine),
        );

        // Adjectives
        self.add_word(
            WordEntry::new("grande", "big", PartOfSpeech::Adjective)
                .with_alt("large")
                .with_note("Same form for masc/fem"),
        );
        self.add_word(
            WordEntry::new("pequeño", "small", PartOfSpeech::Adjective)
                .with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("pequeña", "small", PartOfSpeech::Adjective)
                .with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("bueno", "good", PartOfSpeech::Adjective).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("buena", "good", PartOfSpeech::Adjective).with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("malo", "bad", PartOfSpeech::Adjective).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("mala", "bad", PartOfSpeech::Adjective).with_gender(Gender::Feminine),
        );
        self.add_word(
            WordEntry::new("nuevo", "new", PartOfSpeech::Adjective).with_gender(Gender::Masculine),
        );
        self.add_word(
            WordEntry::new("viejo", "old", PartOfSpeech::Adjective).with_gender(Gender::Masculine),
        );

        // Verbs (infinitives)
        self.add_word(
            WordEntry::new("ser", "to be", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("ser", Tense::Infinitive).irregular())
                .with_note("Used for permanent states, identity, origin"),
        );
        self.add_word(
            WordEntry::new("estar", "to be", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("estar", Tense::Infinitive).irregular())
                .with_note("Used for temporary states, location, conditions"),
        );
        self.add_word(
            WordEntry::new("tener", "to have", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("tener", Tense::Infinitive).irregular()),
        );
        self.add_word(
            WordEntry::new("hacer", "to do", PartOfSpeech::Verb)
                .with_alt("to make")
                .with_conjugation(VerbConjugation::new("hacer", Tense::Infinitive).irregular()),
        );
        self.add_word(
            WordEntry::new("ir", "to go", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("ir", Tense::Infinitive).irregular()),
        );
        self.add_word(
            WordEntry::new("comer", "to eat", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("comer", Tense::Infinitive)),
        );
        self.add_word(
            WordEntry::new("hablar", "to speak", PartOfSpeech::Verb)
                .with_alt("to talk")
                .with_conjugation(VerbConjugation::new("hablar", Tense::Infinitive)),
        );
        self.add_word(
            WordEntry::new("vivir", "to live", PartOfSpeech::Verb)
                .with_conjugation(VerbConjugation::new("vivir", Tense::Infinitive)),
        );

        // Conjugated verbs (common forms)
        self.add_word(
            WordEntry::new("soy", "I am", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ser", Tense::Present)
                    .with_person_number(Person::First, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("eres", "you are", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ser", Tense::Present)
                    .with_person_number(Person::Second, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("es", "is", PartOfSpeech::Verb)
                .with_alt("he/she is")
                .with_conjugation(
                    VerbConjugation::new("ser", Tense::Present)
                        .with_person_number(Person::Third, Number::Singular)
                        .irregular(),
                ),
        );
        self.add_word(
            WordEntry::new("somos", "we are", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ser", Tense::Present)
                    .with_person_number(Person::First, Number::Plural)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("son", "they are", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ser", Tense::Present)
                    .with_person_number(Person::Third, Number::Plural)
                    .irregular(),
            ),
        );

        self.add_word(
            WordEntry::new("estoy", "I am", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("estar", Tense::Present)
                    .with_person_number(Person::First, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("está", "is", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("estar", Tense::Present)
                    .with_person_number(Person::Third, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("están", "they are", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("estar", Tense::Present)
                    .with_person_number(Person::Third, Number::Plural)
                    .irregular(),
            ),
        );

        self.add_word(
            WordEntry::new("tengo", "I have", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("tener", Tense::Present)
                    .with_person_number(Person::First, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("tiene", "has", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("tener", Tense::Present)
                    .with_person_number(Person::Third, Number::Singular)
                    .irregular(),
            ),
        );

        self.add_word(
            WordEntry::new("voy", "I go", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ir", Tense::Present)
                    .with_person_number(Person::First, Number::Singular)
                    .irregular(),
            ),
        );
        self.add_word(
            WordEntry::new("va", "goes", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("ir", Tense::Present)
                    .with_person_number(Person::Third, Number::Singular)
                    .irregular(),
            ),
        );

        self.add_word(
            WordEntry::new("hablo", "I speak", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("hablar", Tense::Present)
                    .with_person_number(Person::First, Number::Singular),
            ),
        );
        self.add_word(
            WordEntry::new("habla", "speaks", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("hablar", Tense::Present)
                    .with_person_number(Person::Third, Number::Singular),
            ),
        );
        self.add_word(
            WordEntry::new("hablamos", "we speak", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("hablar", Tense::Present)
                    .with_person_number(Person::First, Number::Plural),
            ),
        );

        self.add_word(
            WordEntry::new("como", "I eat", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("comer", Tense::Present)
                    .with_person_number(Person::First, Number::Singular),
            ),
        );
        self.add_word(
            WordEntry::new("come", "eats", PartOfSpeech::Verb).with_conjugation(
                VerbConjugation::new("comer", Tense::Present)
                    .with_person_number(Person::Third, Number::Singular),
            ),
        );

        // Prepositions
        self.add_word(
            WordEntry::new("en", "in", PartOfSpeech::Preposition)
                .with_alt("on")
                .with_alt("at"),
        );
        self.add_word(WordEntry::new("de", "of", PartOfSpeech::Preposition).with_alt("from"));
        self.add_word(WordEntry::new("a", "to", PartOfSpeech::Preposition).with_alt("at"));
        self.add_word(WordEntry::new("con", "with", PartOfSpeech::Preposition));
        self.add_word(
            WordEntry::new("por", "for", PartOfSpeech::Preposition)
                .with_alt("by")
                .with_alt("through"),
        );
        self.add_word(
            WordEntry::new("para", "for", PartOfSpeech::Preposition).with_alt("in order to"),
        );
        self.add_word(WordEntry::new("sin", "without", PartOfSpeech::Preposition));

        // Conjunctions
        self.add_word(
            WordEntry::new("y", "and", PartOfSpeech::Conjunction)
                .with_note("Changes to 'e' before words starting with 'i-' or 'hi-'"),
        );
        self.add_word(
            WordEntry::new("o", "or", PartOfSpeech::Conjunction)
                .with_note("Changes to 'u' before words starting with 'o-' or 'ho-'"),
        );
        self.add_word(WordEntry::new("pero", "but", PartOfSpeech::Conjunction));
        self.add_word(WordEntry::new(
            "porque",
            "because",
            PartOfSpeech::Conjunction,
        ));
        self.add_word(
            WordEntry::new("que", "that", PartOfSpeech::Conjunction)
                .with_alt("which")
                .with_alt("who"),
        );

        // Adverbs
        self.add_word(WordEntry::new("muy", "very", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("bien", "well", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("mal", "badly", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("mucho", "much", PartOfSpeech::Adverb).with_alt("a lot"));
        self.add_word(WordEntry::new("poco", "little", PartOfSpeech::Adverb).with_alt("few"));
        self.add_word(WordEntry::new("siempre", "always", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("nunca", "never", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("aquí", "here", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("allí", "there", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("ahora", "now", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("hoy", "today", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("mañana", "tomorrow", PartOfSpeech::Adverb));
        self.add_word(WordEntry::new("ayer", "yesterday", PartOfSpeech::Adverb));

        // Common phrases as single entries
        self.add_word(WordEntry::new(
            "buenos días",
            "good morning",
            PartOfSpeech::Interjection,
        ));
        self.add_word(WordEntry::new(
            "buenas noches",
            "good night",
            PartOfSpeech::Interjection,
        ));
        self.add_word(
            WordEntry::new("gracias", "thank you", PartOfSpeech::Interjection).with_alt("thanks"),
        );
        self.add_word(WordEntry::new(
            "por favor",
            "please",
            PartOfSpeech::Interjection,
        ));
        self.add_word(WordEntry::new("hola", "hello", PartOfSpeech::Interjection).with_alt("hi"));
        self.add_word(WordEntry::new(
            "adiós",
            "goodbye",
            PartOfSpeech::Interjection,
        ));
    }

    fn populate_idioms(&mut self) {
        self.idioms.insert(
            "tener hambre".to_string(),
            ("to be hungry".to_string(), "to have hunger".to_string()),
        );
        self.idioms.insert(
            "tener sed".to_string(),
            ("to be thirsty".to_string(), "to have thirst".to_string()),
        );
        self.idioms.insert(
            "tener frío".to_string(),
            ("to be cold".to_string(), "to have cold".to_string()),
        );
        self.idioms.insert(
            "tener calor".to_string(),
            ("to be hot".to_string(), "to have heat".to_string()),
        );
        self.idioms.insert(
            "tener razón".to_string(),
            ("to be right".to_string(), "to have reason".to_string()),
        );
        self.idioms.insert(
            "hacer falta".to_string(),
            ("to be necessary".to_string(), "to make lack".to_string()),
        );
        self.idioms.insert(
            "dar igual".to_string(),
            ("to not matter".to_string(), "to give equal".to_string()),
        );
        self.idioms.insert(
            "echar de menos".to_string(),
            (
                "to miss (someone)".to_string(),
                "to throw of less".to_string(),
            ),
        );
    }

    fn populate_verb_conjugations(&mut self) {
        // This would contain full conjugation tables
        // For now, individual forms are in words
    }

    fn add_word(&mut self, entry: WordEntry) {
        self.words
            .entry(entry.spanish.to_lowercase())
            .or_default()
            .push(entry);
    }

    /// Look up a word
    #[must_use]
    pub fn lookup(&self, word: &str) -> Option<&Vec<WordEntry>> {
        self.words.get(&word.to_lowercase())
    }

    /// Check if phrase is an idiom
    #[must_use]
    pub fn check_idiom(&self, phrase: &str) -> Option<(&str, &str)> {
        let normalized = phrase.to_lowercase();
        self.idioms
            .get(&normalized)
            .map(|(e, l)| (e.as_str(), l.as_str()))
    }
}

impl Default for SpanishDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Spanish Tutor
// ============================================================================

/// Spanish language tutor
pub struct SpanishTutor {
    dictionary: SpanishDictionary,
}

impl SpanishTutor {
    /// Create new tutor
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: SpanishDictionary::new(),
        }
    }

    /// Translate a word or phrase
    pub fn translate(&self, spanish: &str) -> TranslationResult {
        let normalized = spanish.trim().to_lowercase();

        // Check for idioms first
        if let Some((english, literal)) = self.dictionary.check_idiom(&normalized) {
            let mut result = TranslationResult::new(spanish, english).as_idiom(literal);
            self.add_grammar_for_idiom(&mut result);
            return result;
        }

        // Try as single word/phrase
        if let Some(entries) = self.dictionary.lookup(&normalized) {
            let entry = &entries[0];
            let mut result = TranslationResult::new(spanish, &entry.english.join(" / "));
            result.add_word(entry.clone());
            self.add_grammar_for_word(&mut result, entry);
            return result;
        }

        // Break into words
        let words: Vec<&str> = normalized.split_whitespace().collect();
        let mut english_parts = Vec::new();
        let mut result = TranslationResult::new(spanish, "");

        for word in &words {
            if let Some(entries) = self.dictionary.lookup(word) {
                let entry = &entries[0];
                english_parts.push(entry.english[0].clone());
                result.add_word(entry.clone());
                self.add_grammar_for_word(&mut result, entry);
            } else {
                english_parts.push(format!("[{word}]"));
                result.add_word(WordEntry::new(
                    word,
                    &format!("[unknown: {word}]"),
                    PartOfSpeech::Unknown,
                ));
            }
        }

        result.english = english_parts.join(" ");
        self.check_agreement(&mut result);

        result
    }

    fn add_grammar_for_word(&self, result: &mut TranslationResult, entry: &WordEntry) {
        match entry.pos {
            PartOfSpeech::Verb => {
                if let Some(ref conj) = entry.conjugation {
                    if conj.is_irregular {
                        result.add_grammar(GrammarExplanation::new(
                            "Irregular Verb",
                            &format!(
                                "'{}' is irregular - memorize its conjugations",
                                conj.infinitive
                            ),
                        ));
                    }
                    if matches!(conj.tense, Tense::Present) {
                        result.add_grammar(
                            GrammarExplanation::new(
                                "Present Tense",
                                "Used for current actions, habits, and general truths",
                            )
                            .with_example("Hablo español", "I speak Spanish"),
                        );
                    }
                }
            }
            PartOfSpeech::Article => {
                if let Some(gender) = entry.gender {
                    result.add_grammar(
                        GrammarExplanation::new(
                            "Article Agreement",
                            &format!(
                                "Articles must match noun gender: {} = {}",
                                entry.spanish,
                                if matches!(gender, Gender::Masculine) {
                                    "masculine"
                                } else {
                                    "feminine"
                                }
                            ),
                        )
                        .with_example("el libro", "the book (masc)")
                        .with_example("la casa", "the house (fem)"),
                    );
                }
            }
            PartOfSpeech::Adjective => {
                result.add_grammar(
                    GrammarExplanation::new(
                        "Adjective Placement",
                        "Most adjectives come AFTER the noun in Spanish",
                    )
                    .with_example("el carro rojo", "the red car")
                    .with_example("la casa grande", "the big house"),
                );
                if entry.gender.is_some() {
                    result.add_grammar(
                        GrammarExplanation::new(
                            "Adjective Agreement",
                            "Adjectives must agree in gender and number with the noun",
                        )
                        .with_example("niño alto", "tall boy")
                        .with_example("niña alta", "tall girl"),
                    );
                }
            }
            PartOfSpeech::Noun => {
                if let Some(gender) = entry.gender {
                    result.add_grammar(GrammarExplanation::new(
                        "Noun Gender",
                        &format!(
                            "'{}' is {} - this affects article and adjective agreement",
                            entry.spanish,
                            if matches!(gender, Gender::Masculine) {
                                "masculine"
                            } else {
                                "feminine"
                            }
                        ),
                    ));
                }
            }
            _ => {}
        }
    }

    fn add_grammar_for_idiom(&self, result: &mut TranslationResult) {
        result.add_grammar(GrammarExplanation::new(
            "Idiomatic Expression",
            "This phrase has a meaning different from its literal translation. Learn it as a unit.",
        ));
    }

    fn check_agreement(&self, result: &mut TranslationResult) {
        // Check for gender/number agreement issues
        let mut last_article_gender: Option<Gender> = None;
        let mut needs_agreement_warning = false;

        for entry in &result.word_breakdown {
            if entry.pos == PartOfSpeech::Article {
                last_article_gender = entry.gender;
            } else if entry.pos == PartOfSpeech::Noun || entry.pos == PartOfSpeech::Adjective {
                if let (Some(art_g), Some(word_g)) = (last_article_gender, entry.gender) {
                    if art_g != word_g
                        && !matches!(art_g, Gender::Neuter)
                        && !matches!(word_g, Gender::Neuter)
                    {
                        needs_agreement_warning = true;
                        break;
                    }
                }
            }
        }

        if needs_agreement_warning {
            result.add_grammar(GrammarExplanation::new(
                "⚠️ Agreement Check",
                "Article and noun/adjective genders should match",
            ));
        }
    }

    /// Get dictionary stats
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

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Demo P: Spanish Language Tutor ===\n");

    let tutor = SpanishTutor::new();
    println!("Dictionary loaded: {} entries\n", tutor.dictionary_size());

    let examples = [
        "hola",
        "el libro",
        "la casa grande",
        "yo hablo español",
        "tengo hambre",
        "buenos días",
        "él es bueno",
        "ella está aquí",
    ];

    for spanish in examples {
        println!("{}", "─".repeat(50));
        let result = tutor.translate(spanish);
        println!("{}\n", result.format());
    }

    println!("=== Demo P Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part_of_speech_display() {
        assert_eq!(format!("{}", PartOfSpeech::Noun), "noun");
        assert_eq!(format!("{}", PartOfSpeech::Verb), "verb");
    }

    #[test]
    fn test_gender_display() {
        assert_eq!(format!("{}", Gender::Masculine), "masc");
        assert_eq!(format!("{}", Gender::Feminine), "fem");
    }

    #[test]
    fn test_tense_display() {
        assert_eq!(format!("{}", Tense::Present), "present");
        assert_eq!(format!("{}", Tense::Preterite), "preterite");
    }

    #[test]
    fn test_verb_conjugation_new() {
        let conj = VerbConjugation::new("hablar", Tense::Present);
        assert_eq!(conj.infinitive, "hablar");
        assert!(!conj.is_irregular);
    }

    #[test]
    fn test_verb_conjugation_irregular() {
        let conj = VerbConjugation::new("ser", Tense::Present).irregular();
        assert!(conj.is_irregular);
    }

    #[test]
    fn test_word_entry_new() {
        let entry = WordEntry::new("casa", "house", PartOfSpeech::Noun);
        assert_eq!(entry.spanish, "casa");
        assert_eq!(entry.english[0], "house");
    }

    #[test]
    fn test_word_entry_with_alt() {
        let entry = WordEntry::new("casa", "house", PartOfSpeech::Noun).with_alt("home");
        assert_eq!(entry.english.len(), 2);
    }

    #[test]
    fn test_word_entry_with_gender() {
        let entry =
            WordEntry::new("libro", "book", PartOfSpeech::Noun).with_gender(Gender::Masculine);
        assert_eq!(entry.gender, Some(Gender::Masculine));
    }

    #[test]
    fn test_translation_result_new() {
        let result = TranslationResult::new("hola", "hello");
        assert_eq!(result.spanish, "hola");
        assert_eq!(result.english, "hello");
        assert!(!result.is_idiom);
    }

    #[test]
    fn test_translation_result_idiom() {
        let result =
            TranslationResult::new("tener hambre", "to be hungry").as_idiom("to have hunger");
        assert!(result.is_idiom);
        assert_eq!(
            result.literal_translation,
            Some("to have hunger".to_string())
        );
    }

    #[test]
    fn test_dictionary_new() {
        let dict = SpanishDictionary::new();
        assert!(dict.lookup("hola").is_some());
    }

    #[test]
    fn test_dictionary_lookup() {
        let dict = SpanishDictionary::new();
        let entries = dict.lookup("casa");
        assert!(entries.is_some());
        assert_eq!(entries.unwrap()[0].english[0], "house");
    }

    #[test]
    fn test_dictionary_lookup_case_insensitive() {
        let dict = SpanishDictionary::new();
        assert!(dict.lookup("Casa").is_some());
        assert!(dict.lookup("CASA").is_some());
    }

    #[test]
    fn test_dictionary_idiom() {
        let dict = SpanishDictionary::new();
        let idiom = dict.check_idiom("tener hambre");
        assert!(idiom.is_some());
        let (eng, lit) = idiom.unwrap();
        assert_eq!(eng, "to be hungry");
        assert_eq!(lit, "to have hunger");
    }

    #[test]
    fn test_tutor_new() {
        let tutor = SpanishTutor::new();
        assert!(tutor.dictionary_size() > 0);
    }

    #[test]
    fn test_tutor_translate_single_word() {
        let tutor = SpanishTutor::new();
        let result = tutor.translate("hola");
        assert!(result.english.contains("hello") || result.english.contains("hi"));
    }

    #[test]
    fn test_tutor_translate_phrase() {
        let tutor = SpanishTutor::new();
        let result = tutor.translate("el libro");
        assert!(result.word_breakdown.len() >= 2);
    }

    #[test]
    fn test_tutor_translate_idiom() {
        let tutor = SpanishTutor::new();
        let result = tutor.translate("tener hambre");
        assert!(result.is_idiom);
    }

    #[test]
    fn test_tutor_unknown_word() {
        let tutor = SpanishTutor::new();
        let result = tutor.translate("asdfqwerty");
        // Unknown words are wrapped in brackets in the english output
        assert!(result.english.contains("[asdfqwerty]"));
        // And the word_breakdown contains the unknown marker
        assert!(result
            .word_breakdown
            .iter()
            .any(|e| e.pos == PartOfSpeech::Unknown));
    }

    #[test]
    fn test_grammar_explanation_new() {
        let g = GrammarExplanation::new("Test Rule", "Test explanation");
        assert_eq!(g.rule, "Test Rule");
    }

    #[test]
    fn test_grammar_explanation_with_example() {
        let g = GrammarExplanation::new("Test", "Explanation").with_example("ejemplo", "example");
        assert_eq!(g.examples.len(), 1);
    }

    #[test]
    fn test_word_entry_format() {
        let entry =
            WordEntry::new("casa", "house", PartOfSpeech::Noun).with_gender(Gender::Feminine);
        let formatted = entry.format();
        assert!(formatted.contains("casa"));
        assert!(formatted.contains("house"));
        assert!(formatted.contains("noun"));
    }

    #[test]
    fn test_conjugation_format() {
        let conj = VerbConjugation::new("hablar", Tense::Present)
            .with_person_number(Person::First, Number::Singular);
        let formatted = conj.format();
        assert!(formatted.contains("hablar"));
        assert!(formatted.contains("present"));
        assert!(formatted.contains("1st"));
    }

    #[test]
    fn test_dictionary_has_articles() {
        let dict = SpanishDictionary::new();
        assert!(dict.lookup("el").is_some());
        assert!(dict.lookup("la").is_some());
        assert!(dict.lookup("los").is_some());
        assert!(dict.lookup("las").is_some());
    }

    #[test]
    fn test_dictionary_has_verbs() {
        let dict = SpanishDictionary::new();
        assert!(dict.lookup("ser").is_some());
        assert!(dict.lookup("estar").is_some());
        assert!(dict.lookup("tener").is_some());
    }

    #[test]
    fn test_dictionary_conjugated_verbs() {
        let dict = SpanishDictionary::new();
        let entries = dict.lookup("soy");
        assert!(entries.is_some());
        let entry = &entries.unwrap()[0];
        assert!(entry.conjugation.is_some());
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
            let tutor = SpanishTutor::new();
            let result = tutor.translate(&word);
            prop_assert!(!result.spanish.is_empty());
            prop_assert!(!result.english.is_empty());
        }

        #[test]
        fn prop_dictionary_lookup_consistent(word in "(hola|casa|libro|ser|estar)") {
            let dict = SpanishDictionary::new();
            let r1 = dict.lookup(&word);
            let r2 = dict.lookup(&word);
            prop_assert_eq!(r1.is_some(), r2.is_some());
        }

        #[test]
        fn prop_word_entry_format_not_empty(spanish in "[a-z]+", english in "[a-z]+") {
            let entry = WordEntry::new(&spanish, &english, PartOfSpeech::Noun);
            let formatted = entry.format();
            prop_assert!(!formatted.is_empty());
            prop_assert!(formatted.contains(&spanish));
        }

        #[test]
        fn prop_translation_format_not_empty(text in "[a-z ]{1,20}") {
            let result = TranslationResult::new(&text, "translation");
            let formatted = result.format();
            prop_assert!(!formatted.is_empty());
        }

        #[test]
        fn prop_conjugation_format_contains_infinitive(verb in "(hablar|comer|vivir)") {
            let conj = VerbConjugation::new(&verb, Tense::Present);
            let formatted = conj.format();
            prop_assert!(formatted.contains(&verb));
        }
    }
}
