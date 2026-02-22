//! ALICE-Search × ALICE-Analytics Bridge
//!
//! Search query metrics: unique queries (HLL), query latency (DDSketch),
//! popular patterns (CMS).

use alice_analytics::prelude::*;

/// Search query metrics collector.
pub struct SearchMetrics {
    /// Unique query pattern estimation.
    pub unique_queries: HyperLogLog,
    /// Query latency quantiles.
    pub latency: DDSketch,
    /// Query pattern frequency.
    pub pattern_freq: CountMinSketch,
    /// Total queries.
    pub total: u64,
}

impl SearchMetrics {
    pub fn new() -> Self {
        Self {
            unique_queries: HyperLogLog::new(),
            latency: DDSketch::new(0.01),
            pattern_freq: CountMinSketch::new(),
            total: 0,
        }
    }

    /// Record a search query execution.
    pub fn record_query(&mut self, pattern: &[u8], latency_us: f64) {
        self.unique_queries.insert_bytes(pattern);
        self.latency.insert(latency_us);
        self.pattern_freq.insert_bytes(pattern);
        self.total += 1;
    }

    pub fn unique_query_count(&self) -> f64 {
        self.unique_queries.cardinality()
    }
    pub fn p99_latency(&self) -> f64 {
        self.latency.quantile(0.99)
    }
    pub fn p50_latency(&self) -> f64 {
        self.latency.quantile(0.50)
    }
    pub fn pattern_frequency(&self, pattern: &[u8]) -> u64 {
        self.pattern_freq.estimate_bytes(pattern)
    }
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_metrics() {
        let mut m = SearchMetrics::new();
        for _ in 0..50 {
            m.record_query(b"hello", 100.0);
        }
        m.record_query(b"world", 200.0);
        assert!(m.unique_query_count() >= 1.0);
        assert_eq!(m.total, 51);
    }
}
