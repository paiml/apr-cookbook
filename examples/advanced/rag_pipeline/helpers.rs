#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::HashMap;

pub fn generate_sample_corpus(count: usize, _seed: u64) -> Vec<Document> {
    let topics = [
        "Machine learning is AI.",
        "Neural networks mimic neurons.",
        "Deep learning uses layers.",
        "NLP understands text.",
        "Vision interprets images.",
        "RL trains via rewards.",
        "Transfer learning reuses knowledge.",
        "Attention focuses on relevant info.",
        "Transformers model sequences.",
        "Embeddings are vector spaces.",
    ];
    (0..count)
        .map(|i| {
            let ti = i % topics.len();
            Document::new(
                format!("doc_{i}"),
                format!("{} Doc {i} topic {ti}.", topics[ti]),
            )
            .with_metadata("topic", format!("{ti}"))
            .with_metadata("index", format!("{i}"))
        })
        .collect()
}
