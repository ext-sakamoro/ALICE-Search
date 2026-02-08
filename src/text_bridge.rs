//! ALICE-Search × ALICE-Text Bridge
//!
//! Full-text search over ALICE-Text compressed data.
//! Decompress → Build FM-Index → O(M) pattern search.

use crate::AliceIndex;
use alice_text::{ALICEText, EncodingMode};

/// Search index built from ALICE-Text compressed data.
pub struct CompressedSearchIndex {
    index: AliceIndex,
    decompressed_len: usize,
}

impl CompressedSearchIndex {
    /// Build FM-Index from ALICE-Text compressed bytes.
    ///
    /// Decompresses the text, then builds the search index.
    /// The compressed bytes can be discarded after building.
    pub fn from_compressed(compressed: &[u8], sa_sample_rate: usize) -> Result<Self, String> {
        let mut alice = ALICEText::new(EncodingMode::Pattern);
        let text = alice.decompress(compressed).map_err(|e| format!("{}", e))?;
        let bytes = text.as_bytes();
        let index = AliceIndex::build(bytes, sa_sample_rate);
        Ok(Self {
            index,
            decompressed_len: bytes.len(),
        })
    }

    /// Build FM-Index from plain text, then return compressed form.
    pub fn from_text(text: &str, sa_sample_rate: usize) -> (Self, Vec<u8>) {
        let index = AliceIndex::build(text.as_bytes(), sa_sample_rate);
        let mut alice = ALICEText::new(EncodingMode::Pattern);
        let compressed = alice.compress(text).unwrap_or_default();
        let s = Self {
            index,
            decompressed_len: text.len(),
        };
        (s, compressed)
    }

    /// O(M) pattern count (independent of corpus size).
    pub fn count(&self, pattern: &[u8]) -> usize {
        self.index.count(pattern)
    }

    /// O(M) existence check.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.index.contains(pattern)
    }

    /// Locate all pattern occurrences (zero-allocation iterator).
    pub fn locate(&self, pattern: &[u8]) -> impl Iterator<Item = usize> + '_ {
        self.index.locate(pattern)
    }

    /// Original decompressed text length.
    pub fn text_len(&self) -> usize {
        self.decompressed_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_from_text() {
        let text = "2024-01-15 INFO User logged in\n2024-01-15 INFO User logged out";
        let (idx, compressed) = CompressedSearchIndex::from_text(text, 4);
        assert_eq!(idx.count(b"INFO"), 2);
        assert!(idx.contains(b"logged"));
        assert!(!idx.contains(b"ERROR"));
        assert!(compressed.len() > 0);
    }
}
