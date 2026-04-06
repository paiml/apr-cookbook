//! Chat KV-Cache Management Example
//!
//! Demonstrates key-value cache management for multi-turn conversational
//! inference, where cached attention states from prior turns are reused
//! to avoid redundant computation.
//!
//! # KV-Cache Strategy
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Turn 1: "Hello"        → Compute K,V for 5 tokens, cache them  │
//! │  Turn 2: "How are you?" → Reuse cached K,V + compute 12 new     │
//! │  Turn 3: "Tell me..."   → Reuse all cached + compute 8 new      │
//! │                                                                  │
//! │  Without cache: Recompute ALL tokens every turn (O(n^2))         │
//! │  With cache:    Compute only NEW tokens each turn (O(n))         │
//! │                                                                  │
//! │  Cache eviction: LRU when memory budget exceeded                 │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example chat_kv_cache
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const HIDDEN_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
const NUM_LAYERS: usize = 4;

/// Key-Value pair for a single attention head at one position
#[derive(Clone, Debug)]
#[allow(dead_code)] // Fields used for memory accounting, not direct reads
struct KVEntry {
    key: Vec<f32>,   // HEAD_DIM
    value: Vec<f32>, // HEAD_DIM
}

/// Per-layer KV cache
#[derive(Clone)]
struct LayerCache {
    entries: Vec<KVEntry>, // One per cached position, per head
    num_heads: usize,
}

impl LayerCache {
    fn new(num_heads: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_heads,
        }
    }

    fn len(&self) -> usize {
        self.entries.len() / self.num_heads
    }

    fn memory_bytes(&self) -> usize {
        self.entries.len() * HEAD_DIM * 2 * 4 // key + value, f32
    }

    fn push(&mut self, entries: Vec<KVEntry>) {
        self.entries.extend(entries);
    }

    fn evict_oldest(&mut self, n_positions: usize) {
        let entries_to_remove = n_positions * self.num_heads;
        if entries_to_remove >= self.entries.len() {
            self.entries.clear();
        } else {
            self.entries.drain(..entries_to_remove);
        }
    }
}

/// Full KV cache across all layers
struct KVCache {
    layers: Vec<LayerCache>,
    max_positions: usize,
}

impl KVCache {
    fn new(num_layers: usize, num_heads: usize, max_positions: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerCache::new(num_heads))
                .collect(),
            max_positions,
        }
    }

    fn cached_positions(&self) -> usize {
        self.layers.first().map_or(0, LayerCache::len)
    }

    fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(LayerCache::memory_bytes).sum()
    }

    fn add_positions(&mut self, layer_idx: usize, entries: Vec<KVEntry>) {
        self.layers[layer_idx].push(entries);

        // LRU eviction if over budget
        if self.layers[layer_idx].len() > self.max_positions {
            let excess = self.layers[layer_idx].len() - self.max_positions;
            self.layers[layer_idx].evict_oldest(excess);
        }
    }

    #[cfg(test)]
    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.entries.clear();
        }
    }
}

/// Simple attention model that uses KV cache
struct CachedAttentionModel {
    seed: u64,
    compute_count: usize,
}

impl CachedAttentionModel {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            compute_count: 0,
        }
    }

    /// Compute K,V projections for new tokens at a given layer
    fn compute_kv(&mut self, tokens: &[u32], layer: usize) -> Vec<KVEntry> {
        let mut entries = Vec::with_capacity(tokens.len() * NUM_HEADS);
        for &token in tokens {
            for head in 0..NUM_HEADS {
                let key: Vec<f32> = (0..HEAD_DIM)
                    .map(|d| {
                        let mut h = DefaultHasher::new();
                        (self.seed, "key", layer, head, token, d).hash(&mut h);
                        h.finish() as f32 / u64::MAX as f32 - 0.5
                    })
                    .collect();
                let value: Vec<f32> = (0..HEAD_DIM)
                    .map(|d| {
                        let mut h = DefaultHasher::new();
                        (self.seed, "value", layer, head, token, d).hash(&mut h);
                        h.finish() as f32 / u64::MAX as f32 - 0.5
                    })
                    .collect();
                entries.push(KVEntry { key, value });
                self.compute_count += 1;
            }
        }
        entries
    }

    /// Run inference with KV cache: only compute new tokens
    fn forward_with_cache(&mut self, new_tokens: &[u32], cache: &mut KVCache) -> Vec<f32> {
        for layer in 0..NUM_LAYERS {
            let kv_entries = self.compute_kv(new_tokens, layer);
            cache.add_positions(layer, kv_entries);
        }

        // Simulate final output (hidden states for last token)
        let mut output = vec![0.0f32; HIDDEN_DIM];
        let last_token = new_tokens.last().copied().unwrap_or(0);
        for (i, out) in output.iter_mut().enumerate() {
            let mut h = DefaultHasher::new();
            (self.seed, "output", last_token, cache.cached_positions(), i).hash(&mut h);
            *out = h.finish() as f32 / u64::MAX as f32 - 0.5;
        }
        output
    }

    /// Run inference without cache (recompute everything)
    fn forward_no_cache(&mut self, all_tokens: &[u32]) -> Vec<f32> {
        // Compute all KV pairs from scratch
        for layer in 0..NUM_LAYERS {
            let _ = self.compute_kv(all_tokens, layer);
        }

        let mut output = vec![0.0f32; HIDDEN_DIM];
        let last_token = all_tokens.last().copied().unwrap_or(0);
        for (i, out) in output.iter_mut().enumerate() {
            let mut h = DefaultHasher::new();
            (self.seed, "output", last_token, all_tokens.len(), i).hash(&mut h);
            *out = h.finish() as f32 / u64::MAX as f32 - 0.5;
        }
        output
    }
}

/// Simulate a multi-turn conversation
fn simulate_conversation(turns: &[Vec<u32>], max_cache_positions: usize) -> ConversationStats {
    let mut model_cached = CachedAttentionModel::new(42);
    let mut model_nocache = CachedAttentionModel::new(42);
    let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, max_cache_positions);
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut turn_stats = Vec::new();

    for (turn_idx, turn_tokens) in turns.iter().enumerate() {
        let cached_before = model_cached.compute_count;
        let nocache_before = model_nocache.compute_count;

        // With cache: only compute new tokens
        let _output_cached = model_cached.forward_with_cache(turn_tokens, &mut cache);

        // Without cache: recompute everything
        all_tokens.extend_from_slice(turn_tokens);
        let _output_nocache = model_nocache.forward_no_cache(&all_tokens);

        let cached_ops = model_cached.compute_count - cached_before;
        let nocache_ops = model_nocache.compute_count - nocache_before;

        turn_stats.push(TurnStats {
            turn: turn_idx + 1,
            new_tokens: turn_tokens.len(),
            total_tokens: all_tokens.len(),
            cached_ops,
            nocache_ops,
            cache_positions: cache.cached_positions(),
            cache_memory_kb: cache.total_memory_bytes() as f64 / 1024.0,
            speedup: if cached_ops > 0 {
                nocache_ops as f64 / cached_ops as f64
            } else {
                1.0
            },
        });
    }

    ConversationStats {
        total_cached_ops: model_cached.compute_count,
        total_nocache_ops: model_nocache.compute_count,
        turn_stats,
    }
}

struct TurnStats {
    turn: usize,
    new_tokens: usize,
    total_tokens: usize,
    cached_ops: usize,
    nocache_ops: usize,
    cache_positions: usize,
    cache_memory_kb: f64,
    speedup: f64,
}

struct ConversationStats {
    total_cached_ops: usize,
    total_nocache_ops: usize,
    turn_stats: Vec<TurnStats>,
}

impl ConversationStats {
    fn overall_speedup(&self) -> f64 {
        if self.total_cached_ops == 0 {
            return 1.0;
        }
        self.total_nocache_ops as f64 / self.total_cached_ops as f64
    }
}

fn main() {
    println!("=== Chat KV-Cache Management Example ===\n");

    // =========================================================================
    // Section 1: Multi-Turn Conversation
    // =========================================================================
    println!("1. Multi-Turn Conversation with KV Cache");
    println!("   ─────────────────────────────────────────");

    let turns: Vec<Vec<u32>> = vec![
        vec![1, 8, 5, 12, 12, 15],    // "hello" (6 tokens)
        vec![8, 15, 23, 0, 1, 18, 5], // "how are" (7 tokens)
        vec![20, 5, 12, 12, 0, 13, 5, 0, 1, 0, 19, 20, 15, 18, 25], // longer turn (15 tokens)
        vec![23, 8, 1, 20, 0, 9, 19], // "what is" (7 tokens)
        vec![20, 8, 1, 14, 11, 0, 25, 15, 21], // "thank you" (9 tokens)
    ];

    let stats = simulate_conversation(&turns, 256);

    println!(
        "   {:>5} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "Turn", "New", "Total", "CachedOps", "NoCacheOps", "Cache", "Speedup"
    );
    println!("   {}", "─".repeat(65));

    for ts in &stats.turn_stats {
        println!(
            "   {:>5} {:>8} {:>8} {:>10} {:>10} {:>7}KB {:>7.1}x",
            ts.turn,
            ts.new_tokens,
            ts.total_tokens,
            ts.cached_ops,
            ts.nocache_ops,
            ts.cache_memory_kb as usize,
            ts.speedup
        );
    }
    println!("   {}", "─".repeat(65));
    println!(
        "   Overall: {} cached ops vs {} uncached = {:.1}x speedup",
        stats.total_cached_ops,
        stats.total_nocache_ops,
        stats.overall_speedup()
    );
    println!();

    // =========================================================================
    // Section 2: Cache Size Impact
    // =========================================================================
    println!("2. Cache Size Impact");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>12} {:>12} {:>12} {:>10}",
        "MaxPositions", "CachedOps", "NoCacheOps", "Speedup"
    );
    println!("   {}", "─".repeat(50));

    for max_pos in [8, 16, 32, 64, 128, 256] {
        let stats = simulate_conversation(&turns, max_pos);
        println!(
            "   {:>12} {:>12} {:>12} {:>9.1}x",
            max_pos,
            stats.total_cached_ops,
            stats.total_nocache_ops,
            stats.overall_speedup()
        );
    }
    println!();

    // =========================================================================
    // Section 3: Long Conversation with Eviction
    // =========================================================================
    println!("3. Long Conversation with Cache Eviction");
    println!("   ─────────────────────────────────────────");

    let mut long_turns: Vec<Vec<u32>> = Vec::new();
    for i in 0..20 {
        let len = 5 + (i % 7) * 3;
        let tokens: Vec<u32> = (0..len).map(|j| ((i * 7 + j) % 26 + 1) as u32).collect();
        long_turns.push(tokens);
    }

    let stats = simulate_conversation(&long_turns, 32); // Small cache = frequent eviction

    println!("   20 turns, cache limit=32 positions");
    println!(
        "   {:>5} {:>6} {:>6} {:>8} {:>8}",
        "Turn", "New", "Cached", "MemKB", "Speedup"
    );
    println!("   {}", "─".repeat(40));

    for ts in stats.turn_stats.iter().step_by(4) {
        println!(
            "   {:>5} {:>6} {:>6} {:>7.1} {:>7.1}x",
            ts.turn, ts.new_tokens, ts.cache_positions, ts.cache_memory_kb, ts.speedup
        );
    }
    println!("   Overall speedup: {:.1}x", stats.overall_speedup());
    println!();

    // =========================================================================
    // Section 4: Prefix Sharing Across Conversations
    // =========================================================================
    println!("4. Prefix Sharing (System Prompt Reuse)");
    println!("   ─────────────────────────────────────────");

    let system_prompt: Vec<u32> = vec![25, 15, 21, 0, 1, 18, 5, 0, 1, 0, 8, 5, 12, 16]; // 14 tokens
    let n_conversations = 5;

    let mut total_with_sharing = 0usize;
    let mut total_without_sharing = 0usize;

    for conv_id in 0..n_conversations {
        let user_msg: Vec<u32> = (0..8)
            .map(|j| ((conv_id * 3 + j) % 26 + 1) as u32)
            .collect();

        // With prefix sharing: cache system prompt, only compute user tokens
        let mut model_shared = CachedAttentionModel::new(42);
        let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, 256);
        model_shared.forward_with_cache(&system_prompt, &mut cache);
        let shared_base = model_shared.compute_count;
        model_shared.forward_with_cache(&user_msg, &mut cache);
        let shared_ops = model_shared.compute_count - shared_base;

        // Without: recompute everything
        let mut model_full = CachedAttentionModel::new(42);
        let mut full_tokens = system_prompt.clone();
        full_tokens.extend_from_slice(&user_msg);
        model_full.forward_no_cache(&full_tokens);
        let full_ops = model_full.compute_count;

        total_with_sharing += shared_ops;
        total_without_sharing += full_ops;
    }

    println!(
        "   System prompt: {} tokens, {} conversations",
        system_prompt.len(),
        n_conversations
    );
    println!(
        "   With prefix sharing:    {} KV computations",
        total_with_sharing
    );
    println!(
        "   Without prefix sharing: {} KV computations",
        total_without_sharing
    );
    println!(
        "   Savings: {:.1}%",
        (1.0 - total_with_sharing as f64 / total_without_sharing as f64) * 100.0
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(NUM_LAYERS, NUM_HEADS, 128);
        assert_eq!(cache.cached_positions(), 0);
        assert_eq!(cache.total_memory_bytes(), 0);
    }

    #[test]
    fn test_kv_cache_add_positions() {
        let mut model = CachedAttentionModel::new(42);
        let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, 128);

        let tokens = vec![1, 2, 3];
        model.forward_with_cache(&tokens, &mut cache);

        assert_eq!(cache.cached_positions(), 3);
        assert!(cache.total_memory_bytes() > 0);
    }

    #[test]
    fn test_kv_cache_eviction() {
        let mut model = CachedAttentionModel::new(42);
        let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, 5); // Max 5 positions

        model.forward_with_cache(&vec![1, 2, 3, 4, 5], &mut cache);
        assert_eq!(cache.cached_positions(), 5);

        model.forward_with_cache(&vec![6, 7, 8], &mut cache);
        assert_eq!(cache.cached_positions(), 5); // Evicted oldest 3
    }

    #[test]
    fn test_cache_clear() {
        let mut model = CachedAttentionModel::new(42);
        let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, 128);

        model.forward_with_cache(&vec![1, 2, 3], &mut cache);
        assert!(cache.cached_positions() > 0);

        cache.clear();
        assert_eq!(cache.cached_positions(), 0);
    }

    #[test]
    fn test_cached_fewer_ops_than_uncached() {
        let turns = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8], vec![9, 10, 11, 12]];
        let stats = simulate_conversation(&turns, 128);
        assert!(
            stats.total_cached_ops < stats.total_nocache_ops,
            "Cached ops {} should be less than uncached ops {}",
            stats.total_cached_ops,
            stats.total_nocache_ops
        );
    }

    #[test]
    fn test_speedup_increases_with_turns() {
        let turns = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9],
            vec![10, 11, 12],
            vec![13, 14, 15, 16, 17],
        ];
        let stats = simulate_conversation(&turns, 128);

        // Later turns should have higher speedup
        assert!(stats.turn_stats.len() >= 4);
        let first_speedup = stats.turn_stats[0].speedup;
        let last_speedup = stats.turn_stats[3].speedup;
        // First turn has no speedup (nothing cached yet)
        // Later turns benefit from cache
        assert!(
            last_speedup >= first_speedup,
            "Last speedup {} should be >= first {}",
            last_speedup,
            first_speedup
        );
    }

    #[test]
    fn test_conversation_output_deterministic() {
        let turns = vec![vec![1, 2, 3], vec![4, 5]];
        let s1 = simulate_conversation(&turns, 64);
        let s2 = simulate_conversation(&turns, 64);
        assert_eq!(s1.total_cached_ops, s2.total_cached_ops);
        assert_eq!(s1.total_nocache_ops, s2.total_nocache_ops);
    }

    #[test]
    fn test_layer_cache_memory() {
        let mut lc = LayerCache::new(NUM_HEADS);
        assert_eq!(lc.memory_bytes(), 0);

        lc.push(vec![KVEntry {
            key: vec![0.0; HEAD_DIM],
            value: vec![0.0; HEAD_DIM],
        }]);
        assert!(lc.memory_bytes() > 0);
    }
}
