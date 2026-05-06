//! # Registry Metadata Inverted-Index Search
//!
//! Build per-model term index (lowercased, alphanumeric) → manifest
//! hash list. Query: tokenize, lookup each term, intersect. AND-style
//! query semantics; no scoring (use full-text engine for ranking).
//!
//! Demonstrates the **REG.18** recipe for PMAT-147 (registry round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: classic Salton TF-IDF inverted index.
//!
//! Run with: cargo run --example registry_metadata_search
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok { matches: Vec<String> },
    EmptyQuery,
    EmptyIndex,
}

pub struct SearchIndex {
    by_term: BTreeMap<String, BTreeSet<String>>,
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            by_term: BTreeMap::new(),
        }
    }

    pub fn add(&mut self, manifest_hash: &str, description: &str) {
        for term in tokenize(description) {
            self.by_term
                .entry(term)
                .or_default()
                .insert(manifest_hash.to_string());
        }
    }

    pub fn search(&self, query: &str) -> SearchVerdict {
        if query.is_empty() {
            return SearchVerdict::EmptyQuery;
        }
        if self.by_term.is_empty() {
            return SearchVerdict::EmptyIndex;
        }
        let terms: Vec<String> = tokenize(query).collect();
        if terms.is_empty() {
            return SearchVerdict::EmptyQuery;
        }
        let mut iter = terms.iter();
        let first_term = iter.next().unwrap();
        let mut acc: BTreeSet<String> = self.by_term.get(first_term).cloned().unwrap_or_default();
        for t in iter {
            let next_set: BTreeSet<String> = self.by_term.get(t).cloned().unwrap_or_default();
            acc = acc.intersection(&next_set).cloned().collect();
            if acc.is_empty() {
                break;
            }
        }
        SearchVerdict::Ok {
            matches: acc.into_iter().collect(),
        }
    }

    pub fn term_count(&self) -> usize {
        self.by_term.len()
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_metadata_search")?;

    let mut idx = SearchIndex::new();
    idx.add("hash_a", "Llama-2 7B chat fine-tune");
    idx.add("hash_b", "Mistral 7B base model");
    idx.add("hash_c", "Llama-2 70B base");

    println!("\"Llama base\": {:?}", idx.search("Llama base"));
    println!("\"chat\": {:?}", idx.search("chat"));
    println!("\"absent\": {:?}", idx.search("absent"));
    println!("empty: {:?}", idx.search(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_index() -> SearchIndex {
        let mut idx = SearchIndex::new();
        idx.add("hash_a", "Llama-2 7B chat fine-tune");
        idx.add("hash_b", "Mistral 7B base model");
        idx.add("hash_c", "Llama-2 70B base");
        idx
    }

    #[test]
    fn search_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_term_match() {
        let idx = typical_index();
        let v = idx.search("chat");
        if let SearchVerdict::Ok { matches } = v {
            assert_eq!(matches, vec!["hash_a".to_string()]);
        }
    }

    #[test]
    fn multi_term_intersection() {
        let idx = typical_index();
        let v = idx.search("Llama base");
        if let SearchVerdict::Ok { matches } = v {
            // "Llama" matches hash_a + hash_c; "base" matches hash_b + hash_c.
            assert_eq!(matches, vec!["hash_c".to_string()]);
        }
    }

    #[test]
    fn case_insensitive_match() {
        let idx = typical_index();
        let v = idx.search("LLAMA");
        if let SearchVerdict::Ok { matches } = v {
            assert!(matches.contains(&"hash_a".to_string()));
        }
    }

    #[test]
    fn no_match_empty() {
        let idx = typical_index();
        let v = idx.search("nonexistent");
        if let SearchVerdict::Ok { matches } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn empty_query_rejected() {
        let idx = typical_index();
        assert_eq!(idx.search(""), SearchVerdict::EmptyQuery);
    }

    #[test]
    fn empty_index_rejected() {
        let idx = SearchIndex::new();
        assert_eq!(idx.search("anything"), SearchVerdict::EmptyIndex);
    }

    #[test]
    fn punctuation_query_rejected() {
        let idx = typical_index();
        // ",," tokenizes to no terms.
        assert_eq!(idx.search(",,"), SearchVerdict::EmptyQuery);
    }

    #[test]
    fn term_count_tracks_unique() {
        let idx = typical_index();
        assert!(idx.term_count() > 5);
    }

    #[test]
    fn duplicate_add_no_duplicate_match() {
        let mut idx = SearchIndex::new();
        idx.add("hash_a", "test test");
        if let SearchVerdict::Ok { matches } = idx.search("test") {
            assert_eq!(matches.len(), 1);
        }
    }

    #[test]
    fn intersection_short_circuit_on_zero() {
        let idx = typical_index();
        // Query with one matching term + one missing term → empty.
        let v = idx.search("Llama nonexistent");
        if let SearchVerdict::Ok { matches } = v {
            assert!(matches.is_empty());
        }
    }
}
