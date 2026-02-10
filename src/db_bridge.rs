//! ALICE-DB bridge: FM-Index persistence and search metrics
//!
//! Stores search query metrics (query count, result count, latency)
//! into ALICE-DB time-series for monitoring and optimization.

use alice_db::AliceDB;
use std::io;
use std::path::Path;

/// Search metrics sink backed by ALICE-DB.
pub struct SearchMetricsSink {
    query_count_db: AliceDB,
    result_count_db: AliceDB,
    latency_db: AliceDB,
}

impl SearchMetricsSink {
    /// Open search metrics databases.
    pub fn open<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            query_count_db: AliceDB::open(dir.join("query_count"))?,
            result_count_db: AliceDB::open(dir.join("result_count"))?,
            latency_db: AliceDB::open(dir.join("latency"))?,
        })
    }

    /// Record a search query's metrics.
    pub fn record_query(
        &self,
        timestamp_ms: i64,
        result_count: f32,
        latency_us: f32,
    ) -> io::Result<()> {
        self.query_count_db.put(timestamp_ms, 1.0)?;
        self.result_count_db.put(timestamp_ms, result_count)?;
        self.latency_db.put(timestamp_ms, latency_us)?;
        Ok(())
    }

    /// Query latency history.
    pub fn query_latency(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.latency_db.scan(start, end)
    }

    /// Query result count history.
    pub fn query_results(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.result_count_db.scan(start, end)
    }

    /// Flush all databases.
    pub fn flush(&self) -> io::Result<()> {
        self.query_count_db.flush()?;
        self.result_count_db.flush()?;
        self.latency_db.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_search_metrics_roundtrip() {
        let dir = tempdir().unwrap();
        let sink = SearchMetricsSink::open(dir.path()).unwrap();

        for i in 0..20 {
            sink.record_query(i * 1000, (i % 5) as f32, 50.0 + i as f32)
                .unwrap();
        }
        sink.flush().unwrap();

        let latencies = sink.query_latency(0, 20_000).unwrap();
        assert!(!latencies.is_empty());
    }
}
