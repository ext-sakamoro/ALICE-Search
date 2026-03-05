//! Succinct `BitVector` (Optimized)
//!
//! **Interleaved Memory Layout**: [RankHeader(u64) | Body(8 x u64)]
//! Optimized for L1 Cache Locality. Single fetch rank execution.

extern crate alloc;
use alloc::vec::Vec;

/// 512 bits of body + 64 bits of header = 576 bits per block
/// Fits reasonably well in cache lines (9 * u64 = 72 bytes)
const BLOCK_BITS: usize = 512;
const WORDS_PER_BLOCK: usize = 8;
const BLOCK_STRIDE: usize = WORDS_PER_BLOCK + 1; // 1 Header + 8 Body

#[derive(Clone)]
pub struct BitVector {
    /// Interleaved data: [Rank0, Word0..7, Rank1, Word8..15, ...]
    data: Vec<u64>,
    len: usize,
}

impl BitVector {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Push a bit to the vector.
    /// Header placeholders are written during push, finalized by `build_index()`.
    #[inline]
    pub fn push(&mut self, bit: bool) {
        let bit_idx = self.len % BLOCK_BITS;

        // New block start? Push header placeholder.
        if bit_idx == 0 {
            self.data.push(0);
        }

        let word_offset = bit_idx / 64;
        let bit_offset = bit_idx % 64;

        // Calculate target index in interleaved layout
        let block_base = (self.len / BLOCK_BITS) * BLOCK_STRIDE;
        let target_idx = block_base + 1 + word_offset;

        // Ensure space for the word
        if target_idx >= self.data.len() {
            self.data.push(0);
        }

        if bit {
            self.data[target_idx] |= 1 << bit_offset;
        }

        self.len += 1;
    }

    /// Finalize the index. Must be called after all pushes.
    /// Calculates the Rank Headers in-place.
    pub fn build_index(&mut self) {
        if self.len == 0 {
            return;
        }

        let mut sum = 0usize;
        let num_blocks = self.len.div_ceil(BLOCK_BITS);

        for b in 0..num_blocks {
            let base = b * BLOCK_STRIDE;

            // 1. Write current cumulative rank to header
            self.data[base] = sum as u64;

            // 2. Sum up popcounts in this block for the next header
            let bits_in_block = if b == num_blocks - 1 {
                self.len - b * BLOCK_BITS
            } else {
                BLOCK_BITS
            };
            let words_in_block = bits_in_block.div_ceil(64);

            for w in 0..words_in_block {
                sum += self.data[base + 1 + w].count_ones() as usize;
            }
        }
    }

    /// Access bit at index
    #[inline(always)]
    #[must_use]
    pub fn get(&self, i: usize) -> bool {
        let block = i / BLOCK_BITS;
        let offset = i % BLOCK_BITS;
        let word = offset / 64;
        let bit = offset % 64;

        let idx = block * BLOCK_STRIDE + 1 + word;
        (self.data[idx] >> bit) & 1 != 0
    }

    /// Rank1(i): Count 1s in [0..i)
    /// **Cache Optimized**: Fetches header and body from contiguous memory.
    #[inline(always)]
    #[must_use]
    pub fn rank1(&self, i: usize) -> usize {
        if i == 0 {
            return 0;
        }

        // Clamp to len
        let i = i.min(self.len);

        let block = i / BLOCK_BITS;
        let offset = i % BLOCK_BITS;

        // Handle exact block boundary (offset == 0 means we want full previous blocks)
        if offset == 0 && block > 0 {
            // We want all bits up to this block boundary
            // Get header of the current block (which stores cumulative count up to this point)
            let base = block * BLOCK_STRIDE;
            if base < self.data.len() {
                return self.data[base] as usize;
            }
            // If block doesn't exist, count all bits in previous blocks
            let prev_base = (block - 1) * BLOCK_STRIDE;
            let mut r = self.data[prev_base] as usize;
            for w in 0..WORDS_PER_BLOCK {
                if prev_base + 1 + w < self.data.len() {
                    r += self.data[prev_base + 1 + w].count_ones() as usize;
                }
            }
            return r;
        }

        let base = block * BLOCK_STRIDE;

        // 1. Header Load (Base Rank) - Single cache line with body
        let mut r = self.data[base] as usize;

        // 2. Body Sum (Popcount) - Unrolled for ILP
        let word_idx = offset / 64;
        let bit_idx = offset % 64;

        // Sum full words (max 7 iterations, typically fewer)
        for w in 0..word_idx {
            r += self.data[base + 1 + w].count_ones() as usize;
        }

        // 3. Partial Word
        if bit_idx > 0 && base + 1 + word_idx < self.data.len() {
            let mask = (1u64 << bit_idx) - 1;
            r += (self.data[base + 1 + word_idx] & mask).count_ones() as usize;
        }

        r
    }

    /// Rank0(i): Count 0s in [0..i)
    #[inline(always)]
    #[must_use]
    pub fn rank0(&self, i: usize) -> usize {
        i - self.rank1(i)
    }

    /// Rank(bit, i): Generalized rank query
    #[inline(always)]
    #[must_use]
    pub fn rank(&self, bit: bool, i: usize) -> usize {
        if bit {
            self.rank1(i)
        } else {
            self.rank0(i)
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for BitVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank1_simple() {
        let mut bv = BitVector::new();
        // Push: 1 0 1 1 0 1
        bv.push(true);
        bv.push(false);
        bv.push(true);
        bv.push(true);
        bv.push(false);
        bv.push(true);
        bv.build_index();

        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(1), 1);
        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(3), 2);
        assert_eq!(bv.rank1(4), 3);
        assert_eq!(bv.rank1(5), 3);
        assert_eq!(bv.rank1(6), 4);
    }

    #[test]
    fn test_rank0() {
        let mut bv = BitVector::new();
        // Push: 1 0 1 1 0 1
        bv.push(true);
        bv.push(false);
        bv.push(true);
        bv.push(true);
        bv.push(false);
        bv.push(true);
        bv.build_index();

        assert_eq!(bv.rank0(0), 0);
        assert_eq!(bv.rank0(2), 1);
        assert_eq!(bv.rank0(5), 2);
        assert_eq!(bv.rank0(6), 2);
    }

    #[test]
    fn test_get() {
        let mut bv = BitVector::new();
        bv.push(true);
        bv.push(false);
        bv.push(true);
        bv.build_index();

        assert!(bv.get(0));
        assert!(!bv.get(1));
        assert!(bv.get(2));
    }

    #[test]
    fn test_across_block() {
        let mut bv = BitVector::new();
        // Push 1024 bits (2 blocks of 512 bits each)
        for i in 0..1024 {
            bv.push(i % 3 == 0); // Every 3rd bit is 1
        }
        bv.build_index();

        // Count 1s at position 512 (end of first block)
        let count = bv.rank1(512);
        // 0, 3, 6, ... 510 → 171 ones (0..512, step 3)
        assert_eq!(count, 171);

        // Count 1s at position 1024
        let count2 = bv.rank1(1024);
        // 342 ones total
        assert_eq!(count2, 342);
    }

    #[test]
    fn test_interleaved_layout() {
        let mut bv = BitVector::new();
        // Push exactly 512 bits (1 full block)
        for i in 0..512 {
            bv.push(i % 2 == 0); // Alternating
        }
        bv.build_index();

        // Should have 9 u64s: 1 header + 8 body words
        assert_eq!(bv.data.len(), 9);

        // Header should be 0 (cumulative rank before this block)
        assert_eq!(bv.data[0], 0);

        // Rank at 512 should be 256 (half are 1s)
        assert_eq!(bv.rank1(512), 256);
    }

    #[test]
    fn test_empty_bitvector() {
        let mut bv = BitVector::new();
        bv.build_index();

        assert!(bv.is_empty());
        assert_eq!(bv.len(), 0);
        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank0(0), 0);
    }

    #[test]
    fn test_all_zeros() {
        let mut bv = BitVector::new();
        for _ in 0..64 {
            bv.push(false);
        }
        bv.build_index();

        assert_eq!(bv.rank1(64), 0);
        assert_eq!(bv.rank0(64), 64);
        for i in 0..64 {
            assert!(!bv.get(i));
        }
    }

    #[test]
    fn test_all_ones() {
        let mut bv = BitVector::new();
        for _ in 0..64 {
            bv.push(true);
        }
        bv.build_index();

        assert_eq!(bv.rank1(64), 64);
        assert_eq!(bv.rank0(64), 0);
        for i in 0..64 {
            assert!(bv.get(i));
        }
    }

    #[test]
    fn test_single_bit_true() {
        let mut bv = BitVector::new();
        bv.push(true);
        bv.build_index();

        assert_eq!(bv.len(), 1);
        assert!(bv.get(0));
        assert_eq!(bv.rank1(1), 1);
        assert_eq!(bv.rank0(1), 0);
    }

    #[test]
    fn test_single_bit_false() {
        let mut bv = BitVector::new();
        bv.push(false);
        bv.build_index();

        assert_eq!(bv.len(), 1);
        assert!(!bv.get(0));
        assert_eq!(bv.rank1(1), 0);
        assert_eq!(bv.rank0(1), 1);
    }

    #[test]
    fn test_rank_generic() {
        let mut bv = BitVector::new();
        for i in 0..8 {
            bv.push(i % 2 == 0);
        }
        bv.build_index();

        assert_eq!(bv.rank(true, 4), bv.rank1(4));
        assert_eq!(bv.rank(false, 4), bv.rank0(4));
    }

    #[test]
    fn test_three_blocks() {
        let mut bv = BitVector::new();
        // 3ブロック = 1536ビット
        for i in 0..1536 {
            bv.push(i % 4 == 0); // 4ビットに1つ
        }
        bv.build_index();

        // 1536 / 4 = 384個の1
        assert_eq!(bv.rank1(1536), 384);
        // 1ブロック目境界: 512 / 4 = 128個
        assert_eq!(bv.rank1(512), 128);
        // 2ブロック目境界: 1024 / 4 = 256個
        assert_eq!(bv.rank1(1024), 256);
    }

    #[test]
    fn test_is_empty_after_push() {
        let mut bv = BitVector::new();
        assert!(bv.is_empty());
        bv.push(false);
        assert!(!bv.is_empty());
    }

    #[test]
    fn test_rank0_complement_of_rank1() {
        let mut bv = BitVector::new();
        for i in 0..100 {
            bv.push(i % 3 != 0);
        }
        bv.build_index();

        for i in 0..=100 {
            assert_eq!(
                bv.rank0(i) + bv.rank1(i),
                i,
                "rank0 + rank1 != i at position {i}"
            );
        }
    }
}
