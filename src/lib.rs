#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::too_many_lines
)]

//! # ALICE-Search (Ultra Optimized)
//!
//! **FM-Index based full-text search**
//!
//! > "Searching implies counting. Count(Pattern) -> O(Pattern_Length) independent of Corpus Size."
//!
//! ## Optimized Architecture
//!
//! - **Interleaved `BitVector`**: [Header|Body×8] layout for L1 cache locality
//! - **Wavelet Matrix**: Double-buffered build (2 allocations total)
//! - **Iterator Locate**: Zero-allocation result enumeration
//!
//! ## Performance
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | Build | **O(N)** (SA-IS) | O(N × 1.125) |
//! | Count | **O(M)** | O(1) |
//! | Locate | O(M + occ × step) | **O(1)** (iterator) |
//! | Contains | **O(M)** | O(1) |
//!
//! ## Example
//!
//! ```
//! use alice_search::AliceIndex;
//!
//! let text = b"abracadabra";
//! let index = AliceIndex::build(text, 4);
//!
//! // Count - O(pattern_length), NOT O(text_length)!
//! assert_eq!(index.count(b"abra"), 2);
//! assert_eq!(index.count(b"a"), 5);
//!
//! // Check existence
//! assert!(index.contains(b"cadabra"));
//!
//! // Locate positions (zero-allocation iterator)
//! let positions: Vec<_> = index.locate(b"abra").collect();
//! assert_eq!(positions.len(), 2);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bitvec;
pub mod bwt;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod search;
pub mod wavelet;

pub use search::AliceIndex;

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
#[cfg(feature = "cache")]
pub mod cache_bridge;
#[cfg(feature = "db")]
pub mod db_bridge;
#[cfg(feature = "text")]
pub mod text_bridge;

/// Version
pub const VERSION: &str = "0.2.1-ultra-fried";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 4);

        assert_eq!(index.count(b"abra"), 2);
        assert_eq!(index.count(b"bra"), 2);
        assert_eq!(index.count(b"a"), 5);
        assert_eq!(index.count(b"xyz"), 0);
    }

    #[test]
    fn test_locate() {
        let text = b"abracadabra";
        let index = AliceIndex::build(text, 1);

        let mut positions: Vec<_> = index.locate(b"abra").collect();
        positions.sort_unstable();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions, vec![0, 7]);
    }

    #[test]
    fn test_empty_pattern() {
        let text = b"hello";
        let index = AliceIndex::build(text, 4);

        // Empty pattern matches everything
        assert_eq!(index.count(b""), text.len() + 1);
    }

    #[test]
    fn test_full_text_match() {
        let text = b"exactmatch";
        let index = AliceIndex::build(text, 4);

        assert_eq!(index.count(text), 1);
    }

    #[test]
    fn test_version_not_empty() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_all_same_chars() {
        // 全て同じ文字のテキスト
        let text = b"aaaaaaaaaa";
        let index = AliceIndex::build(text, 2);

        assert_eq!(index.count(b"a"), 10);
        assert_eq!(index.count(b"aa"), 9);
        assert_eq!(index.count(b"aaa"), 8);
        assert_eq!(index.count(b"aaaaaaaaaaa"), 0); // テキスト長超え
    }

    #[test]
    fn test_contains_prefix_suffix_infix() {
        let text = b"abcdefgh";
        let index = AliceIndex::build(text, 4);

        assert!(index.contains(b"abc")); // 先頭
        assert!(index.contains(b"fgh")); // 末尾
        assert!(index.contains(b"cde")); // 中間
        assert!(!index.contains(b"xyz"));
    }

    #[test]
    fn test_multiline_text() {
        let text = b"line one\nline two\nline three";
        let index = AliceIndex::build(text, 4);

        assert_eq!(index.count(b"line"), 3);
        assert_eq!(index.count(b"\n"), 2);
        assert!(index.contains(b"two"));
    }

    #[test]
    fn test_unicode_bytes() {
        // UTF-8マルチバイト文字（バイト列として扱う）
        let text = "こんにちは".as_bytes();
        let index = AliceIndex::build(text, 4);

        assert_eq!(index.count(b""), text.len() + 1);
        assert_eq!(index.text_len(), text.len());
    }

    #[test]
    fn test_locate_correctness() {
        // 見つかった全位置が正しいことを検証
        let text = b"abababab";
        let index = AliceIndex::build(text, 1);

        let mut positions = index.locate_all(b"ab");
        positions.sort_unstable();

        // "ab" は 0, 2, 4, 6 の4箇所
        assert_eq!(positions, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_count_vs_locate_consistency() {
        // count と locate の結果が一致することを確認
        let text = b"mississippi";
        let index = AliceIndex::build(text, 1);

        for pattern in &[b"is" as &[u8], b"ss", b"p", b"ippi", b"xyz"] {
            let c = index.count(pattern);
            let l = index.locate_all(pattern).len();
            assert_eq!(c, l, "count vs locate mismatch for {pattern:?}");
        }
    }

    #[test]
    fn test_large_sample_step() {
        // sample_step がテキスト長より大きい場合
        let text = b"hello";
        let index = AliceIndex::build(text, 1000);

        assert_eq!(index.count(b"hello"), 1);
        assert_eq!(index.count(b"he"), 1);
        let mut positions = index.locate_all(b"hello");
        positions.sort_unstable();
        assert_eq!(positions, vec![0]);
    }
}
