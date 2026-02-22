//! # ALICE-Search (Ultra Optimized)
//!
//! **FM-Index based full-text search**
//!
//! > "Searching implies counting. Count(Pattern) -> O(Pattern_Length) independent of Corpus Size."
//!
//! ## Optimized Architecture
//!
//! - **Interleaved BitVector**: [Header|Body×8] layout for L1 cache locality
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
        positions.sort();
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
}
