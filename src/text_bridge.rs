//! ALICE-Search × ALICE-Text Bridge
//!
//! Full-text search over ALICE-Text compressed data.
//! Decompress → Build FM-Index → O(M) pattern search.

use crate::AliceIndex;
use alice_text::{ALICEText, EncodingMode};

/// One occurrence of a query, with the byte range of the text around it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchHit {
    /// Byte offset of the occurrence
    pub offset: usize,
    /// Start of the context window (`offset - radius`, at least 0)
    pub context_start: usize,
    /// End of the context window (`offset + query length + radius`, at most the text length)
    pub context_end: usize,
}

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
        let alice = ALICEText::new(EncodingMode::Pattern);
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
    pub fn locate<'a>(&'a self, pattern: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
        self.index.locate(pattern)
    }

    /// O(M) count of a text query
    pub fn search_count(&self, query: &str) -> usize {
        self.index.count(query.as_bytes())
    }

    /// Every occurrence of `query`, with the context window of `context_radius` bytes on each side
    /// (clamped to the text)
    pub fn search(&self, query: &str, context_radius: usize) -> Vec<SearchHit> {
        self.index
            .locate(query.as_bytes())
            .map(|offset| SearchHit {
                offset,
                context_start: offset.saturating_sub(context_radius),
                context_end: offset
                    .saturating_add(query.len())
                    .saturating_add(context_radius)
                    .min(self.decompressed_len),
            })
            .collect()
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
        assert!(!compressed.is_empty());
    }

    #[test]
    fn a_str_query_counts_its_occurrences() {
        let text = "ERROR: connection failed\nINFO: retry success\nERROR: timeout";
        let (idx, _) = CompressedSearchIndex::from_text(text, 4);
        assert_eq!(idx.search_count("ERROR"), 2);
        assert_eq!(idx.search_count("INFO"), 1);
        assert_eq!(idx.search_count("MISSING"), 0);
    }

    #[test]
    fn each_hit_carries_its_offset_and_a_context_window_clamped_to_the_text() {
        // oracle: the offsets str::match_indices finds, and the window [offset - r, offset + |q| + r]
        // clamped to [0, len]
        let text = "the quick brown fox jumps over the lazy dog";
        let (idx, _) = CompressedSearchIndex::from_text(text, 4);
        for (query, radius) in [("fox", 5), ("the", 3), ("dog", 10), ("t", 0)] {
            let mut hits = idx.search(query, radius);
            hits.sort_by_key(|h| h.offset);
            let want: Vec<(usize, usize, usize)> = text
                .match_indices(query)
                .map(|(o, _)| {
                    (
                        o,
                        o.saturating_sub(radius),
                        (o + query.len() + radius).min(text.len()),
                    )
                })
                .collect();
            let got: Vec<(usize, usize, usize)> = hits
                .iter()
                .map(|h| (h.offset, h.context_start, h.context_end))
                .collect();
            assert_eq!(got, want, "query {query:?} radius {radius}");
            for h in &hits {
                assert_eq!(&text[h.offset..h.offset + query.len()], query);
            }
        }
    }

    #[test]
    fn an_empty_text_has_no_hits() {
        let (idx, _) = CompressedSearchIndex::from_text("", 4);
        assert_eq!(idx.search_count("anything"), 0);
        assert!(idx.search("anything", 3).is_empty());
        assert_eq!(idx.text_len(), 0);
    }

    #[test]
    fn hits_found_in_compressed_input_match_the_plain_text() {
        let text = "alpha beta alpha gamma alpha";
        let (_, compressed) = CompressedSearchIndex::from_text(text, 4);
        let idx = CompressedSearchIndex::from_compressed(&compressed, 4).unwrap();
        let mut offsets: Vec<usize> = idx.search("alpha", 2).iter().map(|h| h.offset).collect();
        offsets.sort_unstable();
        assert_eq!(offsets, vec![0, 11, 23]);
    }

    #[test]
    fn test_text_len_preserved() {
        let text = "hello world";
        let (idx, _) = CompressedSearchIndex::from_text(text, 4);
        assert_eq!(idx.text_len(), text.len());
    }
}
