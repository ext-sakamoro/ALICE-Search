//! Wavelet Matrix (Optimized)
//!
//! **Zero-Allocation Build**: Uses double-buffering (ping-pong) to avoid
//! allocating vectors during construction.
//! **Interleaved BitVector**: Maximizes cache hits during rank queries.
//!
//! Space: N bytes + 12.5% overhead per layer.

extern crate alloc;
use alloc::vec;
use crate::bitvec::BitVector;

/// 8 layers for 8-bit characters (u8)
const LAYERS: usize = 8;

pub struct WaveletMatrix {
    /// BitVector for each layer (interleaved layout)
    layers: [BitVector; LAYERS],
    /// Number of zeros (Z) in each layer, used for routing
    zeros: [usize; LAYERS],
    /// Length of the text
    len: usize,
}

impl WaveletMatrix {
    /// Build Wavelet Matrix with Double Buffering (Ping-Pong)
    ///
    /// **Optimization**: Allocates only 2 auxiliary buffers of size N,
    /// reused across all 8 layers via `mem::swap`.
    /// No intermediate allocations during layer construction.
    pub fn build(text: &[u8]) -> Self {
        let n = text.len();
        let mut layers: [BitVector; LAYERS] = core::array::from_fn(|_| BitVector::new());
        let mut zeros = [0usize; LAYERS];

        if n == 0 {
            return Self { layers, zeros, len: 0 };
        }

        // Ping-Pong buffers: only 2 allocations for entire build
        let mut current = text.to_vec();
        let mut next = vec![0u8; n];

        // Build 8 layers (MSB to LSB)
        for d in (0..LAYERS).rev() {
            let layer = &mut layers[d];
            let bit_mask = 1u8 << d;

            // Pass 1: Count zeros for split point
            let mut zero_count = 0;
            for &c in current.iter() {
                if (c & bit_mask) == 0 {
                    zero_count += 1;
                }
            }
            zeros[d] = zero_count;

            // Pass 2: Distribute values + build BitVector
            let mut z_ptr = 0;
            let mut o_ptr = zero_count;

            for &c in current.iter() {
                let bit = (c & bit_mask) != 0;
                layer.push(bit);

                if bit {
                    next[o_ptr] = c;
                    o_ptr += 1;
                } else {
                    next[z_ptr] = c;
                    z_ptr += 1;
                }
            }

            layer.build_index();

            // Swap buffers (O(1) pointer swap, no copy)
            core::mem::swap(&mut current, &mut next);
        }

        Self { layers, zeros, len: n }
    }

    /// Get character at position i
    /// O(8) operations - fixed cost regardless of alphabet size
    #[inline]
    pub fn get(&self, mut i: usize) -> u8 {
        let mut c = 0u8;

        for d in (0..LAYERS).rev() {
            let bit = self.layers[d].get(i);
            c |= (bit as u8) << d;

            i = if bit {
                self.zeros[d] + self.layers[d].rank1(i)
            } else {
                self.layers[d].rank0(i)
            };
        }
        c
    }

    /// Rank(c, i): Count occurrences of character c in [0..i)
    /// O(8) operations - independent of text size
    #[inline]
    pub fn rank(&self, c: u8, mut i: usize) -> usize {
        let mut start = 0;

        for d in (0..LAYERS).rev() {
            let bit = (c >> d) & 1 != 0;

            let rank_start = self.layers[d].rank(bit, start);
            let rank_end = self.layers[d].rank(bit, i);

            if bit {
                start = self.zeros[d] + rank_start;
                i = self.zeros[d] + rank_end;
            } else {
                start = rank_start;
                i = rank_end;
            }
        }

        i - start
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_get() {
        let text = b"abracadabra";
        let wm = WaveletMatrix::build(text);

        for (i, &c) in text.iter().enumerate() {
            assert_eq!(wm.get(i), c, "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_wavelet_rank() {
        let text = b"abracadabra";
        let wm = WaveletMatrix::build(text);

        // Count 'a' at various positions
        // "abracadabra" - 'a' at 0, 3, 5, 7, 10
        assert_eq!(wm.rank(b'a', 0), 0);
        assert_eq!(wm.rank(b'a', 1), 1);
        assert_eq!(wm.rank(b'a', 4), 2);
        assert_eq!(wm.rank(b'a', 11), 5);

        // Count 'b' - at 1, 8
        assert_eq!(wm.rank(b'b', 0), 0);
        assert_eq!(wm.rank(b'b', 2), 1);
        assert_eq!(wm.rank(b'b', 11), 2);
    }

    #[test]
    fn test_wavelet_all_same() {
        let text = b"aaaaaaaaaa"; // 10 'a's
        let wm = WaveletMatrix::build(text);

        assert_eq!(wm.rank(b'a', 5), 5);
        assert_eq!(wm.rank(b'a', 10), 10);
        assert_eq!(wm.rank(b'b', 10), 0);
    }

    #[test]
    fn test_wavelet_empty() {
        let text = b"";
        let wm = WaveletMatrix::build(text);

        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn test_wavelet_binary() {
        // Test with binary-like data
        let text: Vec<u8> = (0u16..256).map(|x| x as u8).collect();
        let wm = WaveletMatrix::build(&text);

        for i in 0..256 {
            assert_eq!(wm.get(i), i as u8);
        }

        // Each byte appears exactly once
        for c in 0..=255u8 {
            assert_eq!(wm.rank(c, 256), 1);
        }
    }
}
