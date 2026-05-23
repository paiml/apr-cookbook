//! # apr inspect --vocab — Token Query
//!
//! `apr inspect --vocab <FILE>` shows the tokenizer vocabulary. This
//! recipe builds a query function that takes a token string and returns
//! its id (or token id and reverse mapping). Asserts contract: vocab is
//! a 1:1 (token, id) bijection — no duplicate ids, no duplicate tokens.
//!
//! Demonstrates the **INSPECT.8** recipe for PMAT-109 (apr inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender INSPECT-003 + Hugging Face vocab.json convention
//!
//! Run with: cargo run --example cli_inspect_vocab_token_query
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, PartialEq)]
pub enum VocabVerdict {
    Ok,
    DuplicateToken(String),
    DuplicateId(u32),
}

pub fn validate_vocab<S: std::hash::BuildHasher>(vocab: &HashMap<String, u32, S>) -> VocabVerdict {
    let mut seen_ids: HashSet<u32> = HashSet::new();
    for (token, id) in vocab {
        if !seen_ids.insert(*id) {
            return VocabVerdict::DuplicateId(*id);
        }
        // Reverse-lookup check: every other token maps to a different id.
        for (other_token, other_id) in vocab {
            if other_token != token && other_id == id {
                return VocabVerdict::DuplicateToken(token.clone());
            }
        }
    }
    VocabVerdict::Ok
}

pub fn build_reverse_vocab<S: std::hash::BuildHasher>(
    vocab: &HashMap<String, u32, S>,
) -> BTreeMap<u32, String> {
    vocab.iter().map(|(k, v)| (*v, k.clone())).collect()
}

pub fn lookup_token<S: std::hash::BuildHasher>(
    vocab: &HashMap<String, u32, S>,
    token: &str,
) -> Option<u32> {
    vocab.get(token).copied()
}

pub fn lookup_id(reverse: &BTreeMap<u32, String>, id: u32) -> Option<&String> {
    reverse.get(&id)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_inspect_vocab_token_query")?;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<pad>".into(), 0);
    vocab.insert("<eos>".into(), 1);
    vocab.insert("hello".into(), 17);
    vocab.insert("world".into(), 23);

    println!("validation: {:?}", validate_vocab(&vocab));

    let reverse = build_reverse_vocab(&vocab);
    println!("\ntoken → id queries:");
    for t in ["<pad>", "hello", "world", "missing"] {
        println!("  {t:>10}  →  {:?}", lookup_token(&vocab, t));
    }
    println!("\nid → token queries:");
    for id in [0u32, 17, 99] {
        println!("  {id:>3}  →  {:?}", lookup_id(&reverse, id));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy_vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        v.insert("a".into(), 0);
        v.insert("b".into(), 1);
        v.insert("c".into(), 2);
        v
    }

    #[test]
    fn query_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_vocab_validates() {
        assert_eq!(validate_vocab(&happy_vocab()), VocabVerdict::Ok);
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut v = happy_vocab();
        v.insert("d".into(), 0); // duplicate id
        let r = validate_vocab(&v);
        assert!(matches!(
            r,
            VocabVerdict::DuplicateId(_) | VocabVerdict::DuplicateToken(_)
        ));
    }

    #[test]
    fn lookup_existing_token_returns_id() {
        assert_eq!(lookup_token(&happy_vocab(), "a"), Some(0));
        assert_eq!(lookup_token(&happy_vocab(), "c"), Some(2));
    }

    #[test]
    fn lookup_missing_token_returns_none() {
        assert!(lookup_token(&happy_vocab(), "missing").is_none());
    }

    #[test]
    fn reverse_vocab_round_trips() {
        let v = happy_vocab();
        let r = build_reverse_vocab(&v);
        for (token, id) in &v {
            assert_eq!(r.get(id).map(String::as_str), Some(token.as_str()));
        }
    }

    #[test]
    fn empty_vocab_validates_vacuously() {
        let v: HashMap<String, u32> = HashMap::new();
        assert_eq!(validate_vocab(&v), VocabVerdict::Ok);
    }
}
