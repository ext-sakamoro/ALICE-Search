//! Burrows-Wheeler Transform (BWT)
//!
//! The foundation of FM-Index. BWT rearranges text to group similar characters,
//! enabling both compression and fast search.
//!
//! Suffix array construction uses SA-IS (Suffix Array Induced Sorting),
//! Nong, Zhang & Chan 2009 — O(N) time, O(N) space.

extern crate alloc;
use alloc::vec::Vec;

/// Sentinel character (lexicographically smallest)
pub const SENTINEL: u8 = 0;

// ---------------------------------------------------------------------------
// SA-IS: Suffix Array Induced Sorting  O(N) time · O(N) space
// ---------------------------------------------------------------------------
//
// Terminology
//   S-type suffix i : text[i..] < text[i+1..]  (or i == n, the sentinel)
//   L-type suffix i : text[i..] > text[i+1..]
//   LMS (Left-Most S-type): suffix i is S-type and suffix i-1 is L-type.
//
// High-level steps (mirrors the paper exactly):
//   1. Classify every position as S or L.
//   2. Compute bucket boundaries (head / tail) from character frequencies.
//   3. Place LMS suffixes at the *tail* of their buckets (rough order).
//   4. Induced-sort L-type suffixes from left to right (into bucket heads).
//   5. Induced-sort S-type suffixes from right to left (into bucket tails).
//   6. Compact the now-sorted LMS substrings; if names are not unique recurse.
//   7. Use the recursively-sorted order to place LMS suffixes accurately.
//   8. Repeat induced sort (steps 4-5) for the final suffix array.
//
// The public entry point `build_suffix_array` converts u8 text to u32 and
// calls `sais` which operates on a generic integer alphabet.

/// Build Suffix Array using SA-IS (Suffix Array Induced Sorting).
/// O(N) time and O(N) extra space.
///
/// The returned array has length `text.len() + 1`.  `SA[0]` is always
/// `text.len()` (the virtual sentinel position).
pub fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();

    // Edge cases: empty text or single character.
    if n == 0 {
        return vec![0];
    }
    if n == 1 {
        return vec![1, 0];
    }

    // Convert to u32 alphabet, reserving 0 for the appended sentinel.
    // Original bytes occupy values 1..=256, so alphabet size = 257.
    let mut s: Vec<u32> = Vec::with_capacity(n + 1);
    for &b in text {
        s.push(b as u32 + 1);
    }
    s.push(0); // sentinel — strictly smallest

    let alpha = 257usize; // number of distinct symbols possible
    let mut sa = vec![0usize; n + 1];
    sais(&s, &mut sa, alpha);
    sa
}

// ---------------------------------------------------------------------------
// Core SA-IS implementation
// ---------------------------------------------------------------------------

/// Classify each position in `s` as S-type (true) or L-type (false).
/// Position `n` (the sentinel) is always S-type.
fn classify_sl(s: &[u32]) -> Vec<bool> {
    let n = s.len();
    let mut is_s = vec![false; n];
    // Sentinel is S-type.
    is_s[n - 1] = true;
    if n < 2 {
        return is_s;
    }
    // Scan right to left.
    for i in (0..n - 1).rev() {
        is_s[i] = if s[i] < s[i + 1] {
            true
        } else if s[i] > s[i + 1] {
            false
        } else {
            is_s[i + 1] // same character: inherit from right neighbour
        };
    }
    is_s
}

/// True if position `i` is an LMS suffix (Left-Most S-type).
#[inline(always)]
fn is_lms(is_s: &[bool], i: usize) -> bool {
    i > 0 && is_s[i] && !is_s[i - 1]
}

/// Compute bucket sizes (frequencies) for each symbol.
fn bucket_sizes(s: &[u32], alpha: usize) -> Vec<usize> {
    let mut bkt = vec![0usize; alpha];
    for &c in s {
        bkt[c as usize] += 1;
    }
    bkt
}

/// Fill `head[c]` = index of the first slot in bucket c.
fn bucket_heads(bkt: &[usize]) -> Vec<usize> {
    let mut head = vec![0usize; bkt.len()];
    let mut sum = 0;
    for (i, &b) in bkt.iter().enumerate() {
        head[i] = sum;
        sum += b;
    }
    head
}

/// Fill `tail[c]` = index of the last slot in bucket c (inclusive).
fn bucket_tails(bkt: &[usize]) -> Vec<usize> {
    let mut tail = vec![0usize; bkt.len()];
    let mut sum = 0;
    for (i, &b) in bkt.iter().enumerate() {
        sum += b;
        tail[i] = sum - 1;
    }
    tail
}

/// Step 3 — scatter LMS suffixes into the tails of their buckets.
fn place_lms(s: &[u32], sa: &mut [usize], tail: &mut [usize], is_s: &[bool]) {
    // Sentinel marker: usize::MAX means "empty".
    sa.fill(usize::MAX);
    for i in (0..s.len()).rev() {
        if is_lms(is_s, i) {
            let c = s[i] as usize;
            sa[tail[c]] = i;
            // Saturating sub so we don't wrap on 0 (though bucket won't be empty here).
            tail[c] = tail[c].wrapping_sub(1);
        }
    }
}

/// Step 4 — induced-sort L-type suffixes left-to-right.
fn induce_l(s: &[u32], sa: &mut [usize], head: &mut [usize], is_s: &[bool]) {
    let n = s.len();
    for i in 0..n {
        if sa[i] == usize::MAX {
            continue;
        }
        let j = sa[i];
        if j == 0 {
            continue;
        }
        let p = j - 1;
        if !is_s[p] {
            // p is L-type
            let c = s[p] as usize;
            sa[head[c]] = p;
            head[c] += 1;
        }
    }
}

/// Step 5 — induced-sort S-type suffixes right-to-left.
fn induce_s(s: &[u32], sa: &mut [usize], tail: &mut [usize], is_s: &[bool]) {
    let n = s.len();
    for i in (0..n).rev() {
        if sa[i] == usize::MAX {
            continue;
        }
        let j = sa[i];
        if j == 0 {
            continue;
        }
        let p = j - 1;
        if is_s[p] {
            // p is S-type
            let c = s[p] as usize;
            sa[tail[c]] = p;
            tail[c] = tail[c].wrapping_sub(1);
        }
    }
}

/// Check whether two LMS substrings (starting at `i` and `j` in `s`) are equal.
/// An LMS substring runs from an LMS position up to and including the *next* LMS
/// position (inclusive).
fn lms_substrings_equal(s: &[u32], is_s: &[bool], i: usize, j: usize) -> bool {
    // Both must be LMS (caller ensures this for i==j case).
    let n = s.len();
    let mut k = 0usize;
    loop {
        let ai = i + k < n;
        let aj = j + k < n;
        // Compare characters.
        let ci = if ai { s[i + k] } else { u32::MAX };
        let cj = if aj { s[j + k] } else { u32::MAX };
        if ci != cj {
            return false;
        }
        // After the first position check, see if both reached the next LMS boundary.
        if k > 0 && is_lms(is_s, i + k) && is_lms(is_s, j + k) {
            return true;
        }
        // If one side ended (hit sentinel) but not both, they differ.
        if !ai || !aj {
            return false;
        }
        k += 1;
    }
}

/// Recursive SA-IS on string `s` with alphabet size `alpha`.
/// Result is written into `sa` which must have length `s.len()`.
fn sais(s: &[u32], sa: &mut [usize], alpha: usize) {
    let n = s.len();

    // ---- Base case: n == 1 (only the sentinel) ----
    if n == 1 {
        sa[0] = 0;
        return;
    }

    // ---- Base case: n == 2 ----
    // s = [x, sentinel], sentinel < x always, so SA = [1, 0].
    if n == 2 {
        sa[0] = 1;
        sa[1] = 0;
        return;
    }

    // 1. Classify S / L.
    let is_s = classify_sl(s);

    // 2. Bucket information.
    let bkt = bucket_sizes(s, alpha);

    // 3. Place LMS suffixes at bucket tails.
    {
        let mut tail = bucket_tails(&bkt);
        place_lms(s, sa, &mut tail, &is_s);
    }

    // 4. Induced-sort L-type.
    {
        let mut head = bucket_heads(&bkt);
        induce_l(s, sa, &mut head, &is_s);
    }

    // 5. Induced-sort S-type.
    {
        let mut tail = bucket_tails(&bkt);
        induce_s(s, sa, &mut tail, &is_s);
    }

    // 6. Collect sorted LMS positions and assign compact names.
    //
    // After the two induced sorts, the LMS suffixes appear in sorted order
    // within `sa`. We read them off, name them by comparing adjacent LMS
    // substrings, and build a reduced string `s1` of length = #LMS suffixes.

    // Collect LMS positions in sorted order.
    let lms_sorted: Vec<usize> = sa
        .iter()
        .filter(|&&x| x != usize::MAX && is_lms(&is_s, x))
        .copied()
        .collect();

    // Assign names: equal consecutive LMS substrings get the same name.
    let _num_lms = lms_sorted.len();

    // name_of[i] = compact integer name for the LMS suffix at original position i.
    // We reuse part of `sa` as scratch to avoid extra allocation.
    // Specifically: we need n slots for name_of; we allocate separately
    // (the overall algorithm is still O(N) total allocations).
    let mut name_of = vec![0u32; n];
    let mut current_name = 0u32;
    let mut prev_lms: Option<usize> = None;
    for &pos in &lms_sorted {
        let new_name = match prev_lms {
            None => {
                current_name // first LMS always gets name 0
            }
            Some(prev) => {
                if !lms_substrings_equal(s, &is_s, prev, pos) {
                    current_name += 1;
                }
                current_name
            }
        };
        name_of[pos] = new_name;
        prev_lms = Some(pos);
    }
    let alpha1 = (current_name + 1) as usize; // new alphabet size

    // Build reduced string s1: LMS positions in *text order* (left to right),
    // values = their compact names.
    let lms_positions_textorder: Vec<usize> = (0..n).filter(|&i| is_lms(&is_s, i)).collect();
    // lms_positions_textorder is already in ascending order.

    let s1: Vec<u32> = lms_positions_textorder
        .iter()
        .map(|&i| name_of[i])
        .collect();
    let n1 = s1.len(); // == num_lms

    // 7. Sort reduced problem — recurse only if names are not yet unique.
    let mut sa1 = vec![0usize; n1];
    if alpha1 < n1 {
        // Not all unique: recurse.
        sais(&s1, &mut sa1, alpha1);
    } else {
        // All names unique: directly invert the name array (counting sort).
        // s1[i] is unique for each i, so sa1[s1[i]] = i.
        for (i, &name) in s1.iter().enumerate() {
            sa1[name as usize] = i;
        }
    }

    // sa1 now gives the sorted order of *indices into lms_positions_textorder*.
    // Convert back to original positions.
    let lms_sorted_final: Vec<usize> = sa1
        .iter()
        .map(|&idx| lms_positions_textorder[idx])
        .collect();

    // 8. Final induced sort using accurately ordered LMS suffixes.

    // Re-place LMS suffixes in bucket tails using the correct order.
    {
        let mut tail = bucket_tails(&bkt);
        sa.fill(usize::MAX);
        // Insert in *reverse* of sorted order so that earlier (smaller) LMS
        // suffixes end up at lower tail positions within their bucket.
        for &pos in lms_sorted_final.iter().rev() {
            let c = s[pos] as usize;
            sa[tail[c]] = pos;
            tail[c] = tail[c].wrapping_sub(1);
        }
    }

    // Induced-sort L.
    {
        let mut head = bucket_heads(&bkt);
        induce_l(s, sa, &mut head, &is_s);
    }

    // Induced-sort S.
    {
        let mut tail = bucket_tails(&bkt);
        induce_s(s, sa, &mut tail, &is_s);
    }
}

// ---------------------------------------------------------------------------
// BWT and C-table (unchanged API)
// ---------------------------------------------------------------------------

/// Build BWT from text and suffix array.
/// `BWT[i] = text[SA[i] - 1]` (or SENTINEL if `SA[i] == 0`).
#[inline]
pub fn build_bwt(text: &[u8], sa: &[usize]) -> Vec<u8> {
    let mut bwt = Vec::with_capacity(sa.len());

    for &idx in sa {
        if idx == 0 {
            bwt.push(SENTINEL);
        } else {
            bwt.push(text[idx - 1]);
        }
    }

    bwt
}

/// Build C-Table: `C[c]` = count of characters lexicographically smaller than `c`.
/// Used for LF-mapping in backward search.
#[inline]
pub fn build_c_table(bwt: &[u8]) -> [usize; 256] {
    let mut counts = [0usize; 256];
    let mut c_table = [0usize; 256];

    for &c in bwt {
        counts[c as usize] += 1;
    }

    let mut sum = 0;
    for i in 0..256 {
        c_table[i] = sum;
        sum += counts[i];
    }

    c_table
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: naive O(N log² N) reference implementation for cross-checking.
    // ------------------------------------------------------------------
    fn naive_suffix_array(text: &[u8]) -> Vec<usize> {
        let n = text.len();
        let mut sa: Vec<usize> = (0..=n).collect();
        sa.sort_unstable_by(|&a, &b| {
            let s1 = if a < n { &text[a..] } else { &[] };
            let s2 = if b < n { &text[b..] } else { &[] };
            s1.cmp(s2)
        });
        sa
    }

    // ------------------------------------------------------------------
    // SA-IS correctness: matches naive for known strings
    // ------------------------------------------------------------------

    #[test]
    fn test_suffix_array_banana() {
        let text = b"banana";
        let sa = build_suffix_array(text);
        // SA for "banana\0": [6, 5, 3, 1, 0, 4, 2]
        assert_eq!(sa, vec![6, 5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_sais_matches_naive_banana() {
        let text = b"banana";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_sais_matches_naive_abracadabra() {
        let text = b"abracadabra";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_sais_matches_naive_mississippi() {
        let text = b"mississippi";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_sais_matches_naive_various() {
        for text in &[
            "abcdefgh",
            "hgfedcba",
            "aabbccdd",
            "aaaa",
            "abababab",
            "the quick brown fox",
            "a",
        ] {
            let t = text.as_bytes();
            assert_eq!(
                build_suffix_array(t),
                naive_suffix_array(t),
                "mismatch on {:?}",
                text
            );
        }
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_input() {
        // Empty text: only the sentinel position.
        let sa = build_suffix_array(b"");
        assert_eq!(sa, vec![0]);
    }

    #[test]
    fn test_single_char() {
        // "a\0" -> SA = [1, 0]
        let sa = build_suffix_array(b"a");
        assert_eq!(sa, vec![1, 0]);
        assert_eq!(sa, naive_suffix_array(b"a"));
    }

    #[test]
    fn test_two_chars_ascending() {
        let text = b"ab";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_two_chars_descending() {
        let text = b"ba";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_two_same_chars() {
        let text = b"aa";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_all_same_chars_short() {
        // All-same characters is a stress test for bucket collisions.
        let text = b"aaaa";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_all_same_chars_longer() {
        let text = b"aaaaaaaaaa"; // 10 'a's
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_all_same_chars_32() {
        let text = vec![b'z'; 32];
        assert_eq!(build_suffix_array(&text), naive_suffix_array(&text));
    }

    #[test]
    fn test_binary_alphabet() {
        // Only two distinct characters — maximises LMS collisions.
        let text = b"010101010101";
        assert_eq!(build_suffix_array(text), naive_suffix_array(text));
    }

    #[test]
    fn test_all_bytes_ascending() {
        // 0x01..=0x10 in order.
        let text: Vec<u8> = (1u8..=16).collect();
        assert_eq!(build_suffix_array(&text), naive_suffix_array(&text));
    }

    #[test]
    fn test_all_bytes_descending() {
        let text: Vec<u8> = (1u8..=16).rev().collect();
        assert_eq!(build_suffix_array(&text), naive_suffix_array(&text));
    }

    // ------------------------------------------------------------------
    // SA length invariant
    // ------------------------------------------------------------------

    #[test]
    fn test_sa_length() {
        for len in [0usize, 1, 2, 3, 7, 10, 50] {
            let text: Vec<u8> = (0..len).map(|i| (i % 26 + b'a' as usize) as u8).collect();
            let sa = build_suffix_array(&text);
            assert_eq!(sa.len(), len + 1, "SA length wrong for n={}", len);
        }
    }

    #[test]
    fn test_sa_is_permutation() {
        // Every value in 0..=n must appear exactly once.
        let text = b"abracadabra";
        let n = text.len();
        let sa = build_suffix_array(text);
        let mut seen = vec![false; n + 1];
        for &v in &sa {
            assert!(!seen[v], "duplicate SA entry {}", v);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s), "SA is not a permutation of 0..=n");
    }

    // ------------------------------------------------------------------
    // BWT correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_bwt_banana() {
        let text = b"banana";
        let sa = build_suffix_array(text);
        let bwt = build_bwt(text, &sa);

        // SA=[6,5,3,1,0,4,2], BWT derivation:
        //   idx=6 -> text[5]='a'
        //   idx=5 -> text[4]='n'
        //   idx=3 -> text[2]='n'
        //   idx=1 -> text[0]='b'
        //   idx=0 -> SENTINEL
        //   idx=4 -> text[3]='a'
        //   idx=2 -> text[1]='a'
        // => BWT = [a, n, n, b, SENTINEL, a, a]
        assert_eq!(bwt, vec![b'a', b'n', b'n', b'b', SENTINEL, b'a', b'a']);
    }

    #[test]
    fn test_bwt_contains_sentinel() {
        let text = b"banana";
        let sa = build_suffix_array(text);
        let bwt = build_bwt(text, &sa);
        assert_eq!(bwt.len(), 7);
        assert!(bwt.contains(&SENTINEL));
    }

    #[test]
    fn test_bwt_abracadabra() {
        // Known BWT of "abracadabra": "ard$rcaaaabb" (using $ for sentinel).
        // Our sentinel is 0x00.  We verify length and sentinel presence.
        let text = b"abracadabra";
        let sa = build_suffix_array(text);
        let bwt = build_bwt(text, &sa);
        assert_eq!(bwt.len(), text.len() + 1);
        assert!(bwt.contains(&SENTINEL));
        // The sentinel must appear exactly once.
        assert_eq!(bwt.iter().filter(|&&c| c == SENTINEL).count(), 1);
    }

    // ------------------------------------------------------------------
    // C-table
    // ------------------------------------------------------------------

    #[test]
    fn test_c_table() {
        let bwt = vec![SENTINEL, b'a', b'a', b'b', b'a'];
        let c_table = build_c_table(&bwt);

        // C[SENTINEL] = 0  (nothing is smaller than the sentinel)
        assert_eq!(c_table[SENTINEL as usize], 0);
        // C['a'] = 1  (one sentinel before 'a')
        assert_eq!(c_table[b'a' as usize], 1);
        // C['b'] = 1 + 3 = 4  (sentinel + three 'a's before 'b')
        assert_eq!(c_table[b'b' as usize], 4);
        // Everything from 'c' onward should equal 5 (total length).
        for c in b'c'..=255u8 {
            assert_eq!(c_table[c as usize], 5, "C[{}] wrong", c);
        }
    }
}
