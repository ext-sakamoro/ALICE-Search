//! FM-Index Search Implementation (Optimized)
//!
//! **Architecture**:
//! - `BitVector`: Interleaved memory layout for cache locality
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
/// Count(Pattern) -> `O(Pattern_Length)` independent of Corpus Size.
pub struct AliceIndex {
    /// Wavelet Matrix (stores BWT + Rank support)
    wm: WaveletMatrix,
    /// C-Table: Cumulative counts
    c_table: [usize; 256],
    /// Suffix Array sampling step
    sample_step: usize,
    /// Sampled SA values (compact)
    sa_samples: Vec<usize>,
    /// `BitVector` marking sampled positions (Fast locate!)
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
    /// - Space: O(N * 1.125) for WM + `O(N/sample_step)` for SA samples
    #[must_use]
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

        Self {
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
    #[must_use]
    pub fn count(&self, pattern: &[u8]) -> usize {
        let range = self.backward_search(pattern);
        range.end - range.start
    }

    /// Locate all positions where pattern occurs (Iterator version)
    ///
    /// **Zero Allocation**: Returns a lazy iterator instead of Vec.
    ///
    /// # Complexity
    /// - O(M + occ * `sample_step`) where occ = number of occurrences
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
    #[must_use]
    pub fn locate<'a>(&'a self, pattern: &'a [u8]) -> LocateIter<'a> {
        let range = self.backward_search(pattern);
        LocateIter { index: self, range }
    }

    /// Locate all positions (collecting into Vec for convenience)
    ///
    /// Use `locate()` iterator for zero-allocation queries.
    #[must_use]
    pub fn locate_all(&self, pattern: &[u8]) -> Vec<usize> {
        self.locate(pattern).collect()
    }

    /// Check if pattern exists in text
    #[inline(always)]
    #[must_use]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        !self.backward_search(pattern).is_empty()
    }

    /// Get the range in suffix array for a pattern
    ///
    /// Useful for advanced operations
    #[inline(always)]
    #[must_use]
    pub fn search_range(&self, pattern: &[u8]) -> Range<usize> {
        self.backward_search(pattern)
    }

    /// Resolve SA[i] using LF-mapping walk + `BitVector` check
    /// `O(sample_step)` - No linear scan!
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
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
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
    #[must_use]
    pub const fn sample_step(&self) -> usize {
        self.sample_step
    }

    /// Original text length (excluding sentinel)
    #[must_use]
    pub const fn text_len(&self) -> usize {
        self.wm.len().saturating_sub(1)
    }

    /// Compression ratio: `index_size` / `original_size`
    #[must_use]
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

impl Iterator for LocateIter<'_> {
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

impl ExactSizeIterator for LocateIter<'_> {}

// ============================================================================
// Case-insensitive search
// ============================================================================

/// 大文字小文字を無視するインデックス。
///
/// ビルド時にテキストを小文字正規化するため、
/// 検索も自動的に case-insensitive になる。
pub struct CaseInsensitiveIndex {
    inner: AliceIndex,
}

impl CaseInsensitiveIndex {
    /// 大文字小文字無視インデックスを構築する。
    ///
    /// テキストを ASCII 小文字に正規化してからインデックスを作成する。
    #[must_use]
    pub fn build(text: &[u8], sample_step: usize) -> Self {
        let lower: Vec<u8> = text.iter().map(|&b| ascii_lower(b)).collect();
        Self {
            inner: AliceIndex::build(&lower, sample_step),
        }
    }

    /// パターンの出現回数（case-insensitive）。
    #[must_use]
    pub fn count(&self, pattern: &[u8]) -> usize {
        let lower: Vec<u8> = pattern.iter().map(|&b| ascii_lower(b)).collect();
        self.inner.count(&lower)
    }

    /// パターンが存在するか（case-insensitive）。
    #[must_use]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        let lower: Vec<u8> = pattern.iter().map(|&b| ascii_lower(b)).collect();
        self.inner.contains(&lower)
    }

    /// 全出現位置を返す（case-insensitive）。
    #[must_use]
    pub fn locate_all(&self, pattern: &[u8]) -> Vec<usize> {
        let lower: Vec<u8> = pattern.iter().map(|&b| ascii_lower(b)).collect();
        self.inner.locate_all(&lower)
    }

    /// テキスト長。
    #[must_use]
    pub const fn text_len(&self) -> usize {
        self.inner.text_len()
    }
}

/// ASCII 大文字を小文字に変換する。非 ASCII はそのまま。
#[inline]
const fn ascii_lower(b: u8) -> u8 {
    if b >= b'A' && b <= b'Z' {
        b + 32
    } else {
        b
    }
}

// ============================================================================
// Incremental index (append + rebuild)
// ============================================================================

/// インクリメンタルインデックスビルダー。
///
/// テキストを逐次追加し、`rebuild()` で検索可能なインデックスを再構築する。
/// 差分ビルドではなくフル再構築だが、追加 API を提供することで
/// ユーザーコードがテキスト管理を自前で行う必要をなくす。
pub struct IncrementalIndex {
    /// 蓄積されたテキスト。
    buffer: Vec<u8>,
    /// SA サンプリングステップ。
    sample_step: usize,
    /// 現在のインデックス（`rebuild()` 後に有効）。
    index: Option<AliceIndex>,
}

impl IncrementalIndex {
    /// 新規作成。
    #[must_use]
    pub fn new(sample_step: usize) -> Self {
        Self {
            buffer: Vec::new(),
            sample_step: sample_step.max(1),
            index: None,
        }
    }

    /// テキストを追加する。
    ///
    /// 追加後は `rebuild()` を呼ぶまで検索結果に反映されない。
    pub fn append(&mut self, text: &[u8]) {
        self.buffer.extend_from_slice(text);
    }

    /// 蓄積されたテキストからインデックスを再構築する。
    pub fn rebuild(&mut self) {
        self.index = Some(AliceIndex::build(&self.buffer, self.sample_step));
    }

    /// パターンの出現回数。`rebuild()` 後に有効。
    #[must_use]
    pub fn count(&self, pattern: &[u8]) -> usize {
        self.index.as_ref().map_or(0, |idx| idx.count(pattern))
    }

    /// パターンが存在するか。
    #[must_use]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.index.as_ref().is_some_and(|idx| idx.contains(pattern))
    }

    /// 全出現位置。
    #[must_use]
    pub fn locate_all(&self, pattern: &[u8]) -> Vec<usize> {
        self.index
            .as_ref()
            .map_or_else(Vec::new, |idx| idx.locate_all(pattern))
    }

    /// 蓄積テキストの長さ。
    #[must_use]
    pub const fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// インデックスが構築済みか。
    #[must_use]
    pub const fn is_built(&self) -> bool {
        self.index.is_some()
    }

    /// 蓄積テキストをクリアしてインデックスをリセットする。
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.index = None;
    }
}

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
        positions.sort_unstable();

        assert_eq!(positions.len(), 2);
        assert_eq!(positions, vec![0, 7]);
    }

    #[test]
    fn test_locate_all() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 1);

        let mut positions = index.locate_all(b"abra");
        positions.sort_unstable();

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

    #[test]
    fn test_empty_text() {
        // 空テキストのインデックス構築
        let index = AliceIndex::build(b"", 4);

        assert_eq!(index.count(b"a"), 0);
        assert_eq!(index.count(b""), 1); // センチネルのみ
        assert!(!index.contains(b"a"));
        assert_eq!(index.text_len(), 0);
    }

    #[test]
    fn test_single_char_text() {
        let index = AliceIndex::build(b"a", 1);

        assert_eq!(index.count(b"a"), 1);
        assert_eq!(index.count(b"b"), 0);
        assert_eq!(index.text_len(), 1);
        assert!(index.contains(b"a"));
    }

    #[test]
    fn test_pattern_longer_than_text() {
        let index = AliceIndex::build(b"hi", 1);

        assert_eq!(index.count(b"hello"), 0);
        assert!(!index.contains(b"hello"));
    }

    #[test]
    fn test_locate_empty_pattern_returns_all() {
        let text = b"abc";
        let index = AliceIndex::build(text, 1);

        // 空パターンはテキスト長+1を返す（センチネル含む）
        assert_eq!(index.count(b""), text.len() + 1);
    }

    #[test]
    fn test_search_range_not_found() {
        let index = AliceIndex::build(b"hello", 4);
        let range = index.search_range(b"xyz");
        assert!(range.is_empty());
    }

    #[test]
    fn test_search_range_found() {
        let index = AliceIndex::build(b"abracadabra", 4);
        let range = index.search_range(b"abra");
        assert_eq!(range.end - range.start, 2);
    }

    #[test]
    fn test_text_len() {
        let text = b"hello world";
        let index = AliceIndex::build(text, 4);
        assert_eq!(index.text_len(), text.len());
    }

    #[test]
    fn test_sample_step_respected() {
        let index = AliceIndex::build(b"abracadabra", 8);
        assert_eq!(index.sample_step(), 8);

        let index2 = AliceIndex::build(b"abracadabra", 1);
        assert_eq!(index2.sample_step(), 1);
    }

    #[test]
    fn test_sample_step_zero_clamped_to_one() {
        // sample_step=0 は 1 にクランプされる
        let index = AliceIndex::build(b"abracadabra", 0);
        assert_eq!(index.sample_step(), 1);
    }

    #[test]
    fn test_locate_single_occurrence() {
        let index = AliceIndex::build(b"hello world", 1);
        let mut positions = index.locate_all(b"world");
        positions.sort_unstable();
        assert_eq!(positions, vec![6]);
    }

    #[test]
    fn test_locate_no_occurrence() {
        let index = AliceIndex::build(b"hello world", 1);
        let positions = index.locate_all(b"xyz");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_exact_size_iterator_empty_result() {
        let index = AliceIndex::build(b"hello", 1);
        let iter = index.locate(b"xyz");
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_size_bytes_nonzero() {
        let index = AliceIndex::build(b"abracadabra", 4);
        assert!(index.size_bytes() > 0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_compression_ratio_empty() {
        let index = AliceIndex::build(b"", 4);
        assert_eq!(index.compression_ratio(), 0.0);
    }

    #[test]
    fn test_binary_data() {
        // バイナリデータ（0x00を除く）の検索
        let text: Vec<u8> = (1u8..=50).collect();
        let index = AliceIndex::build(&text, 4);

        // バイト値 0x01 は1回だけ出現
        assert_eq!(index.count(&[0x01]), 1);
        // バイト値 0x31 は1回だけ出現
        assert_eq!(index.count(&[0x31]), 1);
    }

    #[test]
    fn test_locate_iter_size_hint_decrements() {
        // イテレータを消費するたびに size_hint の上限が 1 ずつ減少することを確認する。
        // ExactSizeIterator の契約: len() == remaining items。
        let index = AliceIndex::build(b"abracadabra", 1);
        let mut iter = index.locate(b"a"); // 5箇所マッチ

        assert_eq!(iter.len(), 5);
        iter.next();
        assert_eq!(iter.len(), 4);
        iter.next();
        assert_eq!(iter.len(), 3);
        // 残り3件を消費して exhausted 状態
        iter.next();
        iter.next();
        iter.next();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_count_overlapping_pattern() {
        // "aaaa" の中に "aa" は 3 回重複して出現する (位置 0, 1, 2)。
        // FM-Index の count は重複を正しく数えなければならない。
        let text = b"aaaa";
        let index = AliceIndex::build(text, 1);

        assert_eq!(index.count(b"aa"), 3);
        assert_eq!(index.count(b"aaa"), 2);
        assert_eq!(index.count(b"aaaa"), 1);

        // locate との一致確認
        assert_eq!(index.locate_all(b"aa").len(), 3);
    }

    #[test]
    fn test_search_range_empty_pattern_covers_all() {
        // 空パターンの search_range は [0, wm.len()) = [0, text.len()+1) を返す。
        // これは「全サフィックスが空文字列で始まる」という FM-Index の定義通り。
        let text = b"hello";
        let index = AliceIndex::build(text, 4);
        let range = index.search_range(b"");

        assert_eq!(range.start, 0);
        assert_eq!(range.end, text.len() + 1); // センチネル込み
        assert_eq!(range.end - range.start, index.count(b""));
    }

    #[test]
    fn test_sample_step_variants_give_same_locate() {
        // sample_step が異なっても locate 結果は同一でなければならない。
        // sample_step はメモリと速度のトレードオフで、正確性には影響しない。
        let text = b"mississippi";
        let pattern = b"issi";

        let mut results: Vec<Vec<usize>> = Vec::new();
        for &step in &[1usize, 2, 4, 8] {
            let index = AliceIndex::build(text, step);
            let mut positions = index.locate_all(pattern);
            positions.sort_unstable();
            results.push(positions);
        }

        // 全 sample_step で結果が一致する
        for i in 1..results.len() {
            assert_eq!(
                results[0],
                results[i],
                "locate mismatch between step=1 and step={}",
                1 << i
            );
        }
        // "issi" は "mississippi" に 2 回出現 (位置 1, 4)
        assert_eq!(results[0], vec![1, 4]);
    }

    // ====================================================================
    // CaseInsensitiveIndex テスト
    // ====================================================================

    #[test]
    fn test_case_insensitive_count() {
        let text = b"Hello World HELLO world";
        let index = CaseInsensitiveIndex::build(text, 4);

        assert_eq!(index.count(b"hello"), 2);
        assert_eq!(index.count(b"HELLO"), 2);
        assert_eq!(index.count(b"Hello"), 2);
        assert_eq!(index.count(b"world"), 2);
        assert_eq!(index.count(b"WORLD"), 2);
    }

    #[test]
    fn test_case_insensitive_contains() {
        let text = b"Rust Programming";
        let index = CaseInsensitiveIndex::build(text, 4);

        assert!(index.contains(b"rust"));
        assert!(index.contains(b"RUST"));
        assert!(index.contains(b"programming"));
        assert!(index.contains(b"PROGRAMMING"));
        assert!(!index.contains(b"python"));
    }

    #[test]
    fn test_case_insensitive_locate() {
        let text = b"abABab";
        let index = CaseInsensitiveIndex::build(text, 1);

        let mut positions = index.locate_all(b"ab");
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_case_insensitive_text_len() {
        let text = b"Hello";
        let index = CaseInsensitiveIndex::build(text, 4);
        assert_eq!(index.text_len(), 5);
    }

    #[test]
    fn test_case_insensitive_non_ascii_passthrough() {
        // 非 ASCII バイトはそのまま通過する
        let text = "café".as_bytes();
        let index = CaseInsensitiveIndex::build(text, 4);
        assert!(index.contains(b"caf"));
    }

    #[test]
    fn test_ascii_lower() {
        assert_eq!(ascii_lower(b'A'), b'a');
        assert_eq!(ascii_lower(b'Z'), b'z');
        assert_eq!(ascii_lower(b'a'), b'a');
        assert_eq!(ascii_lower(b'z'), b'z');
        assert_eq!(ascii_lower(b'0'), b'0');
        assert_eq!(ascii_lower(b' '), b' ');
        assert_eq!(ascii_lower(0xFF), 0xFF);
    }

    // ====================================================================
    // IncrementalIndex テスト
    // ====================================================================

    #[test]
    fn test_incremental_empty() {
        let index = IncrementalIndex::new(4);
        assert!(!index.is_built());
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.count(b"a"), 0);
        assert!(!index.contains(b"a"));
        assert!(index.locate_all(b"a").is_empty());
    }

    #[test]
    fn test_incremental_append_and_rebuild() {
        let mut index = IncrementalIndex::new(4);
        index.append(b"hello ");
        index.append(b"world");
        assert_eq!(index.buffer_len(), 11);
        assert!(!index.is_built());

        index.rebuild();
        assert!(index.is_built());
        assert_eq!(index.count(b"hello"), 1);
        assert!(index.contains(b"world"));
    }

    #[test]
    fn test_incremental_rebuild_reflects_new_data() {
        let mut index = IncrementalIndex::new(1);
        index.append(b"aaa");
        index.rebuild();
        assert_eq!(index.count(b"a"), 3);

        index.append(b"aa");
        // rebuild前は古いインデックス
        assert_eq!(index.count(b"a"), 3);

        index.rebuild();
        assert_eq!(index.count(b"a"), 5);
    }

    #[test]
    fn test_incremental_locate() {
        let mut index = IncrementalIndex::new(1);
        index.append(b"abcabc");
        index.rebuild();

        let mut positions = index.locate_all(b"abc");
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 3]);
    }

    #[test]
    fn test_incremental_clear() {
        let mut index = IncrementalIndex::new(4);
        index.append(b"hello");
        index.rebuild();
        assert!(index.is_built());

        index.clear();
        assert!(!index.is_built());
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.count(b"hello"), 0);
    }

    #[test]
    fn test_incremental_sample_step_zero_clamped() {
        let index = IncrementalIndex::new(0);
        // sample_step=0 は 1 にクランプされる
        // 正常に動作することを検証
        assert!(!index.is_built());
    }
}
