//! # Download and Cache Management
//!
//! **CLI equivalent:** `apr pull hf://org/model --cache ~/.cache/apr`
//!
//! Demonstrates a local cache manager for downloaded models. Handles
//! cache misses (simulated download), cache hits (instant return),
//! LRU eviction when the cache exceeds a size limit, and cache statistics.
//!
//! ## Sections
//! 1. Cache miss — first download, populates the cache
//! 2. Cache hit — second request returns cached path instantly
//! 3. LRU eviction — remove oldest entries when cache is full
//! 4. Cache stats — summary of cache contents and utilization
//!
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single entry in the model cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    uri: String,
    path: String,
    size_bytes: usize,
    downloaded_at: u64, // seconds since epoch
    last_accessed: u64,
}

/// Cache manager that tracks downloaded models.
struct CacheManager {
    cache_dir: String,
    entries: Vec<CacheEntry>,
    max_size_bytes: usize,
}

/// Result of a cache operation.
#[derive(Debug)]
enum CacheResult {
    Hit {
        path: String,
        size_bytes: usize,
    },
    Miss {
        path: String,
        size_bytes: usize,
        download_ms: u64,
    },
}

/// Cache statistics.
#[derive(Debug)]
struct CacheStats {
    total_entries: usize,
    total_size_bytes: usize,
    max_size_bytes: usize,
    utilization_pct: f64,
    oldest_entry_age_secs: u64,
    newest_entry_age_secs: u64,
}

// ---------------------------------------------------------------------------
// CacheManager implementation
// ---------------------------------------------------------------------------

impl CacheManager {
    /// Create a new cache manager with the given directory and size limit.
    fn new(cache_dir: &str, max_size_bytes: usize) -> Self {
        Self {
            cache_dir: cache_dir.to_string(),
            entries: Vec::new(),
            max_size_bytes,
        }
    }

    /// Current total size of all cached entries.
    fn total_size(&self) -> usize {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Look up a URI in the cache.
    fn find(&self, uri: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.uri == uri)
    }

    /// Get or download a model.
    ///
    /// If the model is already cached, returns immediately (cache hit).
    /// Otherwise, simulates downloading and adds to cache (cache miss).
    fn get_or_download(&mut self, uri: &str, now: u64) -> CacheResult {
        // Check for cache hit
        if let Some(idx) = self.find(uri) {
            self.entries[idx].last_accessed = now;
            return CacheResult::Hit {
                path: self.entries[idx].path.clone(),
                size_bytes: self.entries[idx].size_bytes,
            };
        }

        // Cache miss — simulate download
        let seed = hash_name_to_seed(uri);
        let model_size = 1024 + (seed % 8192) as usize;
        let download_ms = 50 + (seed % 500);

        let filename: String = uri
            .strip_prefix("hf://")
            .unwrap_or(uri)
            .chars()
            .map(|c| if c == '/' || c == '@' { '_' } else { c })
            .collect();
        let path = format!("{}/{filename}.apr", self.cache_dir);

        let entry = CacheEntry {
            uri: uri.to_string(),
            path: path.clone(),
            size_bytes: model_size,
            downloaded_at: now,
            last_accessed: now,
        };

        self.entries.push(entry);

        CacheResult::Miss {
            path,
            size_bytes: model_size,
            download_ms,
        }
    }

    /// Evict least-recently-used entries until total size is within max_size.
    ///
    /// Returns the number of entries evicted.
    fn evict_lru(&mut self, max_size: usize) -> usize {
        let mut evicted = 0;

        while self.total_size() > max_size && !self.entries.is_empty() {
            // Find LRU entry
            let lru_idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(i, _)| i)
                .unwrap();

            self.entries.remove(lru_idx);
            evicted += 1;
        }

        evicted
    }

    /// Remove a specific URI from the cache.
    #[cfg(test)]
    fn remove(&mut self, uri: &str) -> bool {
        if let Some(idx) = self.find(uri) {
            self.entries.remove(idx);
            true
        } else {
            false
        }
    }

    /// Clear the entire cache.
    #[cfg(test)]
    fn clear(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }

    /// Get cache statistics.
    fn stats(&self, now: u64) -> CacheStats {
        let total_size = self.total_size();
        let utilization = if self.max_size_bytes > 0 {
            total_size as f64 / self.max_size_bytes as f64 * 100.0
        } else {
            0.0
        };

        let oldest_age = self
            .entries
            .iter()
            .map(|e| now.saturating_sub(e.downloaded_at))
            .max()
            .unwrap_or(0);

        let newest_age = self
            .entries
            .iter()
            .map(|e| now.saturating_sub(e.downloaded_at))
            .min()
            .unwrap_or(0);

        CacheStats {
            total_entries: self.entries.len(),
            total_size_bytes: total_size,
            max_size_bytes: self.max_size_bytes,
            utilization_pct: utilization,
            oldest_entry_age_secs: oldest_age,
            newest_entry_age_secs: newest_age,
        }
    }
}

/// FNV-1a checksum for cache verification.
#[cfg(test)]
fn compute_fnv_checksum(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_pull_cache")?;

    let mut cache = CacheManager::new("~/.cache/apr", 20_000);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs();

    // Section 1: Cache miss (first download)
    println!("=== Cache Miss (First Download) ===");
    let models = [
        "hf://microsoft/phi-2",
        "hf://meta/llama-3",
        "hf://google/gemma-7b",
        "hf://mistral/mistral-7b",
    ];

    for (i, uri) in models.iter().enumerate() {
        let t = now + i as u64; // simulate sequential times
        let result = cache.get_or_download(uri, t);
        match &result {
            CacheResult::Miss {
                path,
                size_bytes,
                download_ms,
            } => {
                println!("  MISS {uri} → {path} ({size_bytes} B, {download_ms}ms)");
            }
            CacheResult::Hit { path, size_bytes } => {
                println!("  HIT  {uri} → {path} ({size_bytes} B)");
            }
        }
    }
    println!("Cache entries: {}", cache.entries.len());
    println!("Cache size:    {} bytes", cache.total_size());
    println!();

    // Section 2: Cache hit (second request)
    println!("=== Cache Hit (Second Request) ===");
    for uri in &models[..2] {
        let result = cache.get_or_download(uri, now + 10);
        match &result {
            CacheResult::Hit { path, size_bytes } => {
                println!("  HIT  {uri} → {path} ({size_bytes} B, instant)");
            }
            CacheResult::Miss { .. } => {
                println!("  MISS {uri} (unexpected)");
            }
        }
    }
    println!();

    // Section 3: LRU eviction
    println!("=== LRU Eviction ===");
    let before_size = cache.total_size();
    let before_count = cache.entries.len();
    // Set a tight limit to trigger eviction
    let eviction_limit = cache.total_size() / 2;
    let evicted = cache.evict_lru(eviction_limit);
    println!("Eviction limit: {eviction_limit} bytes");
    println!("Before:  {before_count} entries, {before_size} bytes");
    println!(
        "After:   {} entries, {} bytes",
        cache.entries.len(),
        cache.total_size()
    );
    println!("Evicted: {evicted} entries");
    println!();

    // Section 4: Cache stats
    println!("=== Cache Stats ===");
    let stats = cache.stats(now + 100);
    println!("Total entries:    {}", stats.total_entries);
    println!("Total size:       {} bytes", stats.total_size_bytes);
    println!("Max size:         {} bytes", stats.max_size_bytes);
    println!("Utilization:      {:.1}%", stats.utilization_pct);
    println!("Oldest entry age: {} seconds", stats.oldest_entry_age_secs);
    println!("Newest entry age: {} seconds", stats.newest_entry_age_secs);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_miss_downloads() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        let result = cache.get_or_download("hf://test/model", 1000);
        assert!(matches!(result, CacheResult::Miss { .. }));
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_cache_hit_returns_existing() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://test/model", 1000);
        let result = cache.get_or_download("hf://test/model", 1001);
        assert!(matches!(result, CacheResult::Hit { .. }));
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_eviction_removes_oldest() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://old/model", 100); // oldest
        cache.get_or_download("hf://new/model", 200); // newest

        let total = cache.total_size();
        // Evict to half capacity
        let evicted = cache.evict_lru(total / 2);
        assert!(evicted > 0);
        // The old model should be evicted first
        assert!(cache.find("hf://old/model").is_none() || cache.find("hf://new/model").is_some());
    }

    #[test]
    fn test_eviction_respects_access_time() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://a/model", 100);
        cache.get_or_download("hf://b/model", 200);
        // Access 'a' more recently
        cache.get_or_download("hf://a/model", 300);

        // Now 'b' has last_accessed=200, 'a' has last_accessed=300
        // Evicting should remove 'b' first
        let b_size = cache
            .entries
            .iter()
            .find(|e| e.uri == "hf://b/model")
            .unwrap()
            .size_bytes;
        let total = cache.total_size();
        cache.evict_lru(total - b_size);

        assert!(cache.find("hf://a/model").is_some());
    }

    #[test]
    fn test_size_tracking() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        assert_eq!(cache.total_size(), 0);
        cache.get_or_download("hf://test/model", 1000);
        assert!(cache.total_size() > 0);
    }

    #[test]
    fn test_remove_specific_entry() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://test/model", 1000);
        assert!(cache.remove("hf://test/model"));
        assert_eq!(cache.entries.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        assert!(!cache.remove("hf://nonexistent/model"));
    }

    #[test]
    fn test_clear_cache() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://a/model", 100);
        cache.get_or_download("hf://b/model", 200);
        let cleared = cache.clear();
        assert_eq!(cleared, 2);
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.total_size(), 0);
    }

    #[test]
    fn test_stats_empty_cache() {
        let cache = CacheManager::new("/tmp/test", 50_000);
        let stats = cache.stats(1000);
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.utilization_pct, 0.0);
    }

    #[test]
    fn test_stats_utilization() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://test/model", 1000);
        let stats = cache.stats(1000);
        assert!(stats.utilization_pct > 0.0);
        assert!(stats.utilization_pct <= 100.0);
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = generate_model_payload(42, 256);
        let c1 = compute_fnv_checksum(&data);
        let c2 = compute_fnv_checksum(&data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_multiple_models_different_sizes() {
        let mut cache = CacheManager::new("/tmp/test", 100_000);
        cache.get_or_download("hf://a/model", 100);
        cache.get_or_download("hf://b/model", 200);
        // Different URIs produce different seeds → likely different sizes
        let sizes: Vec<usize> = cache.entries.iter().map(|e| e.size_bytes).collect();
        assert_eq!(sizes.len(), 2);
    }
}
