# ALICE-Search

**FM-Index Based Full-Text Search (Ultra Optimized)**

> "Searching implies counting. Count(Pattern) -> O(Pattern_Length) independent of Corpus Size."

## Mathematical Superiority

Traditional search engines (inverted index) are fast but their **index is larger than the original text** (2-10x).

ALICE-Search uses **FM-Index** to search compressed data without decompression:

| Metric | Inverted Index | FM-Index (ALICE) |
|--------|---------------|------------------|
| Index Size | 2-10x original | **~1.0x** original |
| Search Time | O(log N) | **O(M)** (pattern length only!) |
| Memory per Query | O(results) | **Zero allocation** (Iterator) |

## Optimized Architecture

### Interleaved BitVector

Cache-optimized memory layout. Header and body in contiguous memory for single-fetch rank operations.

```
Traditional:     data[] ──ptr──> [...], blocks[] ──ptr──> [...]
                 (2 cache misses)

Interleaved:     [Rank₀|w₀..w₇|Rank₅₁₂|w₈..w₁₅|...]
                 (1 cache line fetch)
```

### Wavelet Matrix

8-layer structure for O(8) rank operations on any character. Built with **double-buffering** (ping-pong) to minimize allocations.

```
Layer 7 (MSB): [0 1 1 0 1 0 ...]  ← bit 7 of each char
Layer 6:       [1 0 0 1 0 1 ...]  ← bit 6 of each char
...
Layer 0 (LSB): [1 1 0 0 1 1 ...]  ← bit 0 of each char
```

### Backward Search Algorithm

```
Pattern: "ALICE"
Process: E -> C -> I -> L -> A (right to left)
Each step: new_range = C[char] + Rank(char, old_range)
Result: Range in suffix array containing all matches
```

**Key insight**: Each step is O(8) fixed operations, so total is O(8M) regardless of text size N!

## Installation

```toml
[dependencies]
alice-search = "0.2"
```

## Usage

### Basic Search

```rust
use alice_search::AliceIndex;

let text = b"abracadabra";
let index = AliceIndex::build(text, 4);

// Count occurrences - O(pattern_length), NOT O(text_length)!
assert_eq!(index.count(b"abra"), 2);
assert_eq!(index.count(b"a"), 5);

// Check existence
assert!(index.contains(b"cadabra"));

// Locate positions (Zero-allocation Iterator)
for pos in index.locate(b"abra") {
    println!("Found at position: {}", pos);
}

// Or collect into Vec
let positions: Vec<_> = index.locate(b"abra").collect();
// positions contains [0, 7] (unordered)
```

### Zero-Allocation Iteration

```rust
use alice_search::AliceIndex;

let text = b"the quick brown fox jumps over the lazy dog";
let index = AliceIndex::build(text, 4);

// Iterator-based locate: NO heap allocation for results
let mut count = 0;
for pos in index.locate(b"the") {
    count += 1;
    if count >= 10 { break; } // Early termination supported
}

// ExactSizeIterator support
let iter = index.locate(b"o");
println!("Will find {} occurrences", iter.len());
```

### Large Text Search

```rust
use alice_search::AliceIndex;

// 1 million characters
let text: Vec<u8> = (0..1_000_000)
    .map(|i| b"ALICE"[i % 5])
    .collect();

let index = AliceIndex::build(&text, 32);

// Still O(5) = O(pattern_length), not O(1_000_000)!
let count = index.count(b"ALICE");
assert_eq!(count, 200_000);
```

## API Reference

### `AliceIndex`

```rust
impl AliceIndex {
    /// Build index from text
    /// sample_step: SA sampling interval (lower = faster locate, more memory)
    pub fn build(text: &[u8], sample_step: usize) -> Self;

    /// Count pattern occurrences - O(M)
    pub fn count(&self, pattern: &[u8]) -> usize;

    /// Check if pattern exists - O(M)
    pub fn contains(&self, pattern: &[u8]) -> bool;

    /// Locate positions (Zero-allocation Iterator) - O(M + occ × step)
    pub fn locate(&self, pattern: &[u8]) -> LocateIter;

    /// Locate positions (Vec version) - O(M + occ × step)
    pub fn locate_all(&self, pattern: &[u8]) -> Vec<usize>;

    /// Get suffix array range for pattern
    pub fn search_range(&self, pattern: &[u8]) -> Range<usize>;

    /// Approximate index size in bytes
    pub fn size_bytes(&self) -> usize;

    /// Original text length
    pub fn text_len(&self) -> usize;

    /// Index size / text size ratio
    pub fn compression_ratio(&self) -> f64;
}
```

### `LocateIter`

```rust
impl Iterator for LocateIter<'_> {
    type Item = usize;  // Position in original text
}

impl ExactSizeIterator for LocateIter<'_> {}
```

## Optimized Specs

### Memory Layout (Interleaved BitVector)

```rust
// 72 bytes per 512-bit block (9 × u64)
// [RankHeader | Word₀ | Word₁ | ... | Word₇]
//     8 bytes   8×8 = 64 bytes

struct BitVector {
    data: Vec<u64>,  // Interleaved: [Rank, Body×8, Rank, Body×8, ...]
    len: usize,
}
```

### Wavelet Matrix Build (Double Buffering)

```rust
// Only 2 allocations for entire 8-layer construction
let mut buf1 = text.to_vec();  // Allocation 1
let mut buf2 = vec![0u8; n];   // Allocation 2

for layer in (0..8).rev() {
    // ... distribute to buf2 ...
    core::mem::swap(&mut buf1, &mut buf2);  // O(1) pointer swap
}
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build | O(N log² N)* | O(N) |
| Count | **O(M)** | O(1) |
| Locate | O(M + occ × step) | **O(1)** (iterator) |
| Contains | **O(M)** | O(1) |

*Use SA-IS algorithm for O(N) construction in production

### Index Size Breakdown

| Component | Size | Purpose |
|-----------|------|---------|
| Wavelet Matrix | ~N × 1.125 bytes | 8 interleaved BitVectors |
| C-Table | 2 KB | Character cumulative counts |
| SA Sample Bits | ~N / 8 bytes | Sampled position markers |
| SA Samples | N / step × 8 bytes | Position lookup values |

## Future Optimizations

1. **SIMD Rank**: AVX2/NEON `popcnt` for parallel bit counting
2. **SA-IS Construction**: O(N) suffix array building
3. **Run-Length BWT**: Further compress repetitive text
4. **ALICE-Zip Integration**: Search compressed archives without extraction

## Integration with ALICE Ecosystem

| Component | Use Case |
|-----------|----------|
| ALICE-DB | Full-text search in content-addressed storage |
| ALICE-Zip | Search inside compressed archives |
| ALICE-Sync | Fast pattern matching in P2P sync |
| ALICE-Crypto | Encrypted searchable index |

## Cross-Crate Bridges

### DB Bridge (feature: `db`)

Search query metrics persistence via [ALICE-DB](../ALICE-DB). Records query count, result count, and latency as time-series data.

```toml
[dependencies]
alice-search = { path = "../ALICE-Search", features = ["db"] }
```

### Cache Bridge (feature: `cache`)

Search result caching via [ALICE-Cache](../ALICE-Cache). Caches FM-Index lookup results keyed by FNV-1a query hash for instant repeated lookups.

```toml
[dependencies]
alice-search = { path = "../ALICE-Search", features = ["cache"] }
```

## License

**GNU AGPLv3** (Affero General Public License v3.0)

This library is free software under the terms of the GNU Affero General Public License.

**Why AGPL?**
Search algorithms are public knowledge, but this **highly optimized Rust implementation** is protected. If you use this to provide a search service, you must release your source code.

For proprietary/commercial use, please contact:
**https://extoria.co.jp/en**

## Author

Moroya Sakamoto

---

*"1 billion characters or 10 characters - the search time is the same. That's mathematics."*
