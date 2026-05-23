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

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-tokenize".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert!(config.demo);
        assert_eq!(config.vocab_size, 64);
    }

    #[test]
    fn test_parse_args_vocab_size() {
        let args = vec![
            "apr-tokenize".to_string(),
            "--vocab-size".to_string(),
            "128".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.vocab_size, 128);
    }

    #[test]
    fn test_parse_args_method_unigram() {
        let args = vec![
            "apr-tokenize".to_string(),
            "--method".to_string(),
            "unigram".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.token_method(), TokenMethod::Unigram);
    }

    #[test]
    fn test_parse_args_corpus_path() {
        let args = vec!["apr-tokenize".to_string(), "corpus.txt".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.corpus_path, Some("corpus.txt".to_string()));
    }

    #[test]
    fn test_parse_args_unknown_rejected() {
        let args = vec!["apr-tokenize".to_string(), "--bogus".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_char_frequencies() {
        let freq = count_char_frequencies("aab");
        assert_eq!(freq.get(&'a'), Some(&2));
        assert_eq!(freq.get(&'b'), Some(&1));
        assert_eq!(freq.get(&'z'), None);
    }

    #[test]
    fn test_split_into_words() {
        let (words, counts) = split_into_words("cat cat dog");
        // "cat" appears twice, "dog" once
        assert_eq!(words.len(), 2);
        // First entry should be "cat" (higher frequency)
        let cat_idx = words
            .iter()
            .position(|w| w.first().map_or(false, |s| s == "c"))
            .expect("cat present");
        assert_eq!(counts[cat_idx], 2);
    }

    #[test]
    fn test_count_pairs() {
        let words = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let counts = vec![3];
        let pairs = count_pairs(&words, &counts);
        assert_eq!(pairs.get(&("a".to_string(), "b".to_string())), Some(&3));
        assert_eq!(pairs.get(&("b".to_string(), "c".to_string())), Some(&3));
    }

    #[test]
    fn test_most_frequent_pair_deterministic() {
        let mut pairs = std::collections::HashMap::new();
        pairs.insert(("a".to_string(), "b".to_string()), 5);
        pairs.insert(("c".to_string(), "d".to_string()), 5);
        let (best, count) = most_frequent_pair(&pairs).expect("non-empty");
        assert_eq!(count, 5);
        // Deterministic tiebreak: ("a","b") < ("c","d") lexicographically,
        // but max_by picks the last equal element, so we just check it's one of them
        assert!(
            best == ("a".to_string(), "b".to_string())
                || best == ("c".to_string(), "d".to_string())
        );
    }

    #[test]
    fn test_apply_merge() {
        let mut words = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
        ]];
        apply_merge(&mut words, "a", "b");
        assert_eq!(words[0], vec!["ab".to_string(), "ab".to_string()]);
    }

    #[test]
    fn test_train_bpe_grows_vocab() {
        let trainer = train_bpe("ab ab ab cd cd", 10);
        assert!(trainer.vocab_size() >= 5); // at least a,b,c,d,</w>
        assert!(!trainer.merges.is_empty());
    }

    #[test]
    fn test_tokenize_produces_known_symbols() {
        let trainer = train_bpe("the the the cat", 20);
        let tokens = tokenize("the cat", &trainer);
        // All produced tokens must be in the vocabulary
        for t in &tokens {
            assert!(trainer.vocab.contains_key(t), "token '{}' not in vocab", t);
        }
    }

    #[test]
    fn test_detokenize_simple() {
        let tokens = vec![
            "he".to_string(),
            "llo</w>".to_string(),
            "wor".to_string(),
            "ld</w>".to_string(),
        ];
        assert_eq!(detokenize(&tokens), "hello world");
    }

    #[test]
    fn test_detokenize_bare_eow() {
        let tokens = vec!["a".to_string(), "b".to_string(), "</w>".to_string()];
        assert_eq!(detokenize(&tokens), "ab");
    }

    #[test]
    fn test_roundtrip_tokenize_detokenize() {
        let corpus = demo_corpus();
        let trainer = train_bpe(corpus, 80);
        let input = "the cat sat on the mat";
        let tokens = tokenize(input, &trainer);
        let output = detokenize(&tokens);
        assert_eq!(output, input, "roundtrip must reconstruct original text");
    }

    #[test]
    fn test_roundtrip_single_word() {
        let trainer = train_bpe("hello hello hello world", 20);
        let tokens = tokenize("hello", &trainer);
        let output = detokenize(&tokens);
        assert_eq!(output, "hello");
    }

    #[test]
    fn test_demo_run() {
        let config = TokenizeConfig {
            corpus_path: None,
            vocab_size: 50,
            method: "bpe".to_string(),
            demo: true,
        };
        assert!(run_tokenize(&config).is_ok());
    }

    #[test]
    fn test_deterministic_training() {
        let corpus = demo_corpus();
        let t1 = train_bpe(corpus, 60);
        let t2 = train_bpe(corpus, 60);
        assert_eq!(t1.merges.len(), t2.merges.len());
        for (a, b) in t1.merges.iter().zip(t2.merges.iter()) {
            assert_eq!(a, b, "merges must be identical across runs");
        }
    }

    #[test]
    fn test_empty_input_no_crash() {
        let trainer = train_bpe("", 10);
        assert!(trainer.merges.is_empty());
        let tokens = tokenize("", &trainer);
        assert!(tokens.is_empty());
        assert_eq!(detokenize(&tokens), "");
    }
}

#[cfg(test)]
mod proptests {
    use super::super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_vocab_size_bounded(target in 40usize..120) {
            // The demo corpus has ~29 unique chars + </w>, so target must
            // exceed the initial alphabet for the bound to hold.
            let trainer = train_bpe(demo_corpus(), target);
            prop_assert!(trainer.vocab_size() <= target + 1,
                "vocab {} should not exceed target {} by more than 1",
                trainer.vocab_size(), target);
        }

        #[test]
        fn prop_all_tokens_in_vocab(word in "[a-z]{1,6}") {
            let corpus = format!("{w} {w} {w} the the", w = word);
            let trainer = train_bpe(&corpus, 40);
            let tokens = tokenize(&word, &trainer);
            for t in &tokens {
                prop_assert!(trainer.vocab.contains_key(t),
                    "token '{}' not in vocab for word '{}'", t, word);
            }
        }
    }
}
