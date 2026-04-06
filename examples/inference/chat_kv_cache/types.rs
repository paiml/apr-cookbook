#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const HIDDEN_DIM: usize = 32;
pub const NUM_HEADS: usize = 4;
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
pub const NUM_LAYERS: usize = 4;

/// Key-Value pair for a single attention head at one position
#[derive(Clone, Debug)]
#[allow(dead_code)] // Fields used for memory accounting, not direct reads
pub struct KVEntry {
    pub key: Vec<f32>,   // HEAD_DIM
    pub value: Vec<f32>, // HEAD_DIM
}

/// Per-layer KV cache
#[derive(Clone)]
pub struct LayerCache {
    pub entries: Vec<KVEntry>, // One per cached position, per head
    pub num_heads: usize,
}

impl LayerCache {
    pub fn new(num_heads: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_heads,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() / self.num_heads
    }

    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * HEAD_DIM * 2 * 4 // key + value, f32
    }

    pub fn push(&mut self, entries: Vec<KVEntry>) {
        self.entries.extend(entries);
    }

    pub fn evict_oldest(&mut self, n_positions: usize) {
        let entries_to_remove = n_positions * self.num_heads;
        if entries_to_remove >= self.entries.len() {
            self.entries.clear();
        } else {
            self.entries.drain(..entries_to_remove);
        }
    }
}

/// Full KV cache across all layers
pub struct KVCache {
    pub layers: Vec<LayerCache>,
    pub max_positions: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_heads: usize, max_positions: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerCache::new(num_heads))
                .collect(),
            max_positions,
        }
    }

    pub fn cached_positions(&self) -> usize {
        self.layers.first().map_or(0, LayerCache::len)
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(LayerCache::memory_bytes).sum()
    }

    pub fn add_positions(&mut self, layer_idx: usize, entries: Vec<KVEntry>) {
        self.layers[layer_idx].push(entries);

        // LRU eviction if over budget
        if self.layers[layer_idx].len() > self.max_positions {
            let excess = self.layers[layer_idx].len() - self.max_positions;
            self.layers[layer_idx].evict_oldest(excess);
        }
    }

    #[cfg(test)]
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.entries.clear();
        }
    }
}

/// Simple attention model that uses KV cache
pub struct CachedAttentionModel {
    pub seed: u64,
    pub compute_count: usize,
}

impl CachedAttentionModel {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            compute_count: 0,
        }
    }

    /// Compute K,V projections for new tokens at a given layer
    pub fn compute_kv(&mut self, tokens: &[u32], layer: usize) -> Vec<KVEntry> {
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
    pub fn forward_with_cache(&mut self, new_tokens: &[u32], cache: &mut KVCache) -> Vec<f32> {
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
    pub fn forward_no_cache(&mut self, all_tokens: &[u32]) -> Vec<f32> {
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
pub fn simulate_conversation(turns: &[Vec<u32>], max_cache_positions: usize) -> ConversationStats {
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

pub struct TurnStats {
    pub turn: usize,
    pub new_tokens: usize,
    pub total_tokens: usize,
    pub cached_ops: usize,
    pub nocache_ops: usize,
    pub cache_positions: usize,
    pub cache_memory_kb: f64,
    pub speedup: f64,
}

pub struct ConversationStats {
    pub total_cached_ops: usize,
    pub total_nocache_ops: usize,
    pub turn_stats: Vec<TurnStats>,
}

impl ConversationStats {
    pub fn overall_speedup(&self) -> f64 {
        if self.total_cached_ops == 0 {
            return 1.0;
        }
        self.total_nocache_ops as f64 / self.total_cached_ops as f64
    }
}
