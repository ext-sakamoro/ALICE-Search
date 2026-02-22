//! FM-Index Search Implementation (Optimized)
//!
//! **Architecture**:
//! - BitVector: Interleaved memory layout for cache locality
//! - Wavelet Matrix: Double-buffered build (zero intermediate allocations)
//! - Locate: Iterator-based (zero allocation for query results)
//!
//! **Core Algorithm**: Backward Search
//! - Process pattern from right to left
//! - Use LF-mapping: `new_pos = C[c] + Rank(c, old_pos)`
//! - Complexity: O(M) where M = pattern length
//!
//! This is independent of text size N. Mathematical victory.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Range;

use crate::bitvec::BitVector;
use crate::bwt::{build_c_table, build_suffix_array, SENTINEL};
use crate::wavelet::WaveletMatrix;

/// ALICE-Search Index (FM-Index implementation)
///
/// Searching implies counting.
/// Count(Pattern) -> O(Pattern_Length) independent of Corpus Size.
pub struct AliceIndex {
    /// Wavelet Matrix (stores BWT + Rank support)
    wm: WaveletMatrix,
    /// C-Table: Cumulative counts
    c_table: [usize; 256],
    /// Suffix Array sampling step
    sample_step: usize,
    /// Sampled SA values (compact)
    sa_samples: Vec<usize>,
    /// BitVector marking sampled positions (Fast locate!)
    sa_sampled_bits: BitVector,
}

impl AliceIndex {
    /// Build index from text
    ///
    /// # Arguments
    /// - `text`: Input text to index
    /// - `sample_step`: SA sampling interval (trade-off: lower = faster locate, more memory)
    ///
    /// # Complexity
    /// - Time: O(N log^2 N) with naive SA (use SA-IS for O(N) in production)
    /// - Space: O(N * 1.125) for WM + O(N/sample_step) for SA samples
    pub fn build(text: &[u8], sample_step: usize) -> Self {
        let sample_step = sample_step.max(1);

        // 1. Build SA & BWT
        let sa = build_suffix_array(text);

        // Reconstruct BWT string for WM construction
        let mut bwt = Vec::with_capacity(sa.len());
        for &idx in &sa {
            if idx == 0 {
                bwt.push(SENTINEL);
            } else {
                bwt.push(text[idx - 1]);
            }
        }

        // 2. Build Wavelet Matrix (Double-buffered, zero intermediate allocs)
        let wm = WaveletMatrix::build(&bwt);
        let c_table = build_c_table(&bwt);

        // 3. Build SA Samples with BitVector
        let mut sa_samples = Vec::new();
        let mut sa_sampled_bits = BitVector::new();

        for &pos in &sa {
            if pos % sample_step == 0 {
                sa_samples.push(pos);
                sa_sampled_bits.push(true);
            } else {
                sa_sampled_bits.push(false);
            }
        }
        sa_sampled_bits.build_index();

        AliceIndex {
            wm,
            c_table,
            sample_step,
            sa_samples,
            sa_sampled_bits,
        }
    }

    /// Count occurrences of a pattern in O(M) time
    ///
    /// M = pattern length. N = text size. **Independent of N!**
    ///
    /// # Example
    /// ```
    /// use alice_search::AliceIndex;
    ///
    /// let index = AliceIndex::build(b"abracadabra", 4);
    /// assert_eq!(index.count(b"abra"), 2);
    /// ```
    #[inline(always)]
    pub fn count(&self, pattern: &[u8]) -> usize {
        let range = self.backward_search(pattern);
        range.end - range.start
    }

    /// Locate all positions where pattern occurs (Iterator version)
    ///
    /// **Zero Allocation**: Returns a lazy iterator instead of Vec.
    ///
    /// # Complexity
    /// - O(M + occ * sample_step) where occ = number of occurrences
    /// - **No heap allocation** for results
    ///
    /// # Example
    /// ```
    /// use alice_search::AliceIndex;
    ///
    /// let index = AliceIndex::build(b"abracadabra", 1);
    /// let positions: Vec<_> = index.locate(b"abra").collect();
    /// assert_eq!(positions.len(), 2);
    /// ```
    #[inline(always)]
    pub fn locate<'a>(&'a self, pattern: &'a [u8]) -> LocateIter<'a> {
        let range = self.backward_search(pattern);
        LocateIter { index: self, range }
    }

    /// Locate all positions (collecting into Vec for convenience)
    ///
    /// Use `locate()` iterator for zero-allocation queries.
    pub fn locate_all(&self, pattern: &[u8]) -> Vec<usize> {
        self.locate(pattern).collect()
    }

    /// Check if pattern exists in text
    #[inline(always)]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        !self.backward_search(pattern).is_empty()
    }

    /// Get the range in suffix array for a pattern
    ///
    /// Useful for advanced operations
    #[inline(always)]
    pub fn search_range(&self, pattern: &[u8]) -> Range<usize> {
        self.backward_search(pattern)
    }

    /// Resolve SA[i] using LF-mapping walk + BitVector check
    /// O(sample_step) - No linear scan!
    fn resolve_sa(&self, mut i: usize) -> usize {
        let mut steps = 0;

        loop {
            // 1. Check if sampled (O(1) with interleaved BitVector)
            if self.sa_sampled_bits.get(i) {
                // Find index in samples vector using rank1 (O(1))
                let idx = self.sa_sampled_bits.rank1(i);
                return self.sa_samples[idx] + steps;
            }

            // 2. Walk backwards (LF-mapping)
            let c = self.wm.get(i);
            if c == SENTINEL {
                return steps; // Hit the start of text
            }

            let rank = self.wm.rank(c, i);
            i = self.c_table[c as usize] + rank;
            steps += 1;
        }
    }

    /// Backward Search Algorithm (FM-Index Core)
    ///
    /// Returns the range [sp, ep) in the suffix array where
    /// all suffixes starting with `pattern` are located.
    #[inline(always)]
    fn backward_search(&self, pattern: &[u8]) -> Range<usize> {
        if pattern.is_empty() {
            return 0..self.wm.len();
        }

        let mut sp = 0;
        let mut ep = self.wm.len();

        // Process pattern from last char to first (backward)
        for &c in pattern.iter().rev() {
            if c == SENTINEL {
                return 0..0;
            }
            let c_idx = c as usize;

            // WM Rank is O(8) [fixed 8 steps for u8]
            let rank_sp = self.wm.rank(c, sp);
            let rank_ep = self.wm.rank(c, ep);

            sp = self.c_table[c_idx] + rank_sp;
            ep = self.c_table[c_idx] + rank_ep;

            if sp >= ep {
                return 0..0; // Pattern not found
            }
        }
        sp..ep
    }

    /// Index size in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let n = self.wm.len();

        // WM: 8 layers × (N/8 bytes for data + N/64 × 8 bytes for blocks)
        // With interleaved layout: 9 u64 per 512 bits = 72 bytes per 512 bits
        // = 1.125 bytes per bit × 8 layers = 9 bytes per character
        let wm_size = n * 9 / 8 * 8; // Approximate

        // C-Table: 256 × sizeof(usize) = 2KB on 64-bit
        let c_table_size = 256 * core::mem::size_of::<usize>();

        // SA sampled bits: interleaved layout
        let sa_bits_size = (n / 512 + 1) * 72; // 72 bytes per block

        // SA samples: (N/step) × sizeof(usize)
        let sa_samples_size = self.sa_samples.len() * core::mem::size_of::<usize>();

        wm_size + c_table_size + sa_bits_size + sa_samples_size
    }

    /// Get the SA sampling step
    #[inline]
    pub fn sample_step(&self) -> usize {
        self.sample_step
    }

    /// Original text length (excluding sentinel)
    pub fn text_len(&self) -> usize {
        self.wm.len().saturating_sub(1)
    }

    /// Compression ratio: index_size / original_size
    pub fn compression_ratio(&self) -> f64 {
        let text_len = self.text_len();
        if text_len == 0 {
            return 0.0;
        }
        let inv_len = 1.0 / text_len as f64;
        self.size_bytes() as f64 * inv_len
    }
}

/// Iterator for locate results.
/// **Zero Allocation** - does not allocate memory for results.
pub struct LocateIter<'a> {
    index: &'a AliceIndex,
    range: Range<usize>,
}

impl<'a> Iterator for LocateIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.range.start >= self.range.end {
            return None;
        }
        let pos = self.index.resolve_sa(self.range.start);
        self.range.start += 1;
        Some(pos)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.range.end - self.range.start;
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for LocateIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_search() {
        let text = b"mississippi";
        let index = AliceIndex::build(text, 4);

        // "issi" appears twice
        assert_eq!(index.count(b"issi"), 2);

        // "mississippi" appears once
        assert_eq!(index.count(b"mississippi"), 1);

        // "xyz" doesn't appear
        assert_eq!(index.count(b"xyz"), 0);
    }

    #[test]
    fn test_count_single_char() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 4);

        assert_eq!(index.count(b"a"), 5);
        assert_eq!(index.count(b"b"), 2);
        assert_eq!(index.count(b"r"), 2);
        assert_eq!(index.count(b"c"), 1);
        assert_eq!(index.count(b"d"), 1);
        assert_eq!(index.count(b"z"), 0);
    }

    #[test]
    fn test_contains() {
        let text = b"hello world";
        let index = AliceIndex::build(text, 4);

        assert!(index.contains(b"hello"));
        assert!(index.contains(b"world"));
        assert!(index.contains(b"o w"));
        assert!(!index.contains(b"xyz"));
    }

    #[test]
    fn test_locate_iterator() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 1);

        // Use iterator (zero allocation)
        let mut positions: Vec<_> = index.locate(b"abra").collect();
        positions.sort();

        assert_eq!(positions.len(), 2);
        assert_eq!(positions, vec![0, 7]);
    }

    #[test]
    fn test_locate_all() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 1);

        let mut positions = index.locate_all(b"abra");
        positions.sort();

        assert_eq!(positions.len(), 2);
        assert!(positions.contains(&0));
        assert!(positions.contains(&7));
    }

    #[test]
    fn test_locate_iterator_exact_size() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 1);

        let iter = index.locate(b"a");
        assert_eq!(iter.len(), 5); // ExactSizeIterator
    }

    #[test]
    fn test_compression_ratio() {
        // Use larger text for realistic ratio
        let mut text = Vec::new();
        for _ in 0..500 {
            text.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
        }
        let index = AliceIndex::build(&text, 32);

        let ratio = index.compression_ratio();
        // With optimizations, ratio should be reasonable
        assert!(ratio > 0.0);
        assert!(ratio < 15.0); // Interleaved layout has some overhead
    }

    #[test]
    fn test_large_text() {
        // Create repetitive text
        let mut text = Vec::new();
        for _ in 0..100 {
            text.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }

        let index = AliceIndex::build(&text, 8);

        assert_eq!(index.count(b"the"), 200);
        assert_eq!(index.count(b"fox"), 100);
        assert_eq!(index.count(b"xyz"), 0);
    }
}
