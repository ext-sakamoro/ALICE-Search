//! ALICE-Cache bridge: Search result caching
//!
//! Caches FM-Index search results using ALICE-Cache to avoid
//! redundant index traversals for repeated queries.

use alice_cache::AliceCache;

/// Cached search result.
#[derive(Clone, Debug)]
pub struct CachedResult {
    /// Byte offsets of matches in the original text.
    pub positions: Vec<usize>,
    /// Number of matches.
    pub count: usize,
}

/// Search result cache backed by ALICE-Cache.
///
/// Keys are FNV-1a hashes of query strings; values are cached
/// position lists from previous FM-Index lookups.
pub struct SearchCache {
    cache: AliceCache<u64, CachedResult>,
}

/// FNV-1a hash for query strings (fast, good distribution).
#[inline]
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl SearchCache {
    /// Create a new search result cache.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: AliceCache::new(capacity),
        }
    }

    /// Look up cached results for a query pattern.
    pub fn get(&self, pattern: &[u8]) -> Option<CachedResult> {
        let key = fnv1a(pattern);
        self.cache.get(&key)
    }

    /// Store search results for a query pattern.
    pub fn put(&self, pattern: &[u8], positions: Vec<usize>) {
        let key = fnv1a(pattern);
        let count = positions.len();
        self.cache.put(key, CachedResult { positions, count });
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_roundtrip() {
        let cache = SearchCache::new(256);
        let pattern = b"hello";
        let positions = vec![10, 42, 99];

        cache.put(pattern, positions.clone());
        let result = cache.get(pattern).unwrap();
        assert_eq!(result.positions, positions);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_cache_miss() {
        let cache = SearchCache::new(256);
        assert!(cache.get(b"missing").is_none());
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a(b"test"), fnv1a(b"test"));
        assert_ne!(fnv1a(b"test"), fnv1a(b"tset"));
    }
}
