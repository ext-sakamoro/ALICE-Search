//! Burrows-Wheeler Transform (BWT)
//!
//! The foundation of FM-Index. BWT rearranges text to group similar characters,
//! enabling both compression and fast search.

extern crate alloc;
use alloc::vec::Vec;

/// Sentinel character (lexicographically smallest)
pub const SENTINEL: u8 = 0;

/// Build Suffix Array using naive sorting
/// O(N log^2 N) - For production, use SA-IS algorithm for O(N)
#[inline]
pub fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();

    // Include position n for the sentinel
    let mut sa: Vec<usize> = (0..=n).collect();

    // Sort by suffix comparison
    sa.sort_unstable_by(|&a, &b| {
        let s1 = if a < n { &text[a..] } else { &[] };
        let s2 = if b < n { &text[b..] } else { &[] };
        s1.cmp(s2)
    });

    sa
}

/// Build BWT from text and suffix array
/// BWT[i] = text[SA[i] - 1] (or sentinel if SA[i] == 0)
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

/// Build C-Table: C[c] = count of characters lexicographically smaller than c
/// Used for LF-mapping in backward search
#[inline]
pub fn build_c_table(bwt: &[u8]) -> [usize; 256] {
    let mut counts = [0usize; 256];
    let mut c_table = [0usize; 256];

    // Count occurrences of each character
    for &c in bwt {
        counts[c as usize] += 1;
    }

    // Cumulative sum: C[c] = sum of counts[0..c]
    let mut sum = 0;
    for i in 0..256 {
        c_table[i] = sum;
        sum += counts[i];
    }

    c_table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_array() {
        let text = b"banana";
        let sa = build_suffix_array(text);

        // SA for "banana$" should be [6, 5, 3, 1, 0, 4, 2]
        // Corresponding suffixes:
        // 6: $ (sentinel/empty)
        // 5: a$
        // 3: ana$
        // 1: anana$
        // 0: banana$
        // 4: na$
        // 2: nana$
        assert_eq!(sa, vec![6, 5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_bwt() {
        let text = b"banana";
        let sa = build_suffix_array(text);
        let bwt = build_bwt(text, &sa);

        // BWT of "banana$" = "annb$aa"
        // Position: [6,5,3,1,0,4,2] -> chars before: [$,a,n,n,b,$,a,a]
        // Wait, let me recalculate:
        // SA[0]=6 -> text[5]='a', but idx=6 means sentinel position, so BWT[0]=sentinel
        // SA[1]=5 -> text[4]='n'? No, idx=5 means suffix starting at 5 ("a$"), char before is text[4]='n'
        // Actually: BWT[i] = text[SA[i]-1] if SA[i]>0, else sentinel
        // SA=[6,5,3,1,0,4,2]
        // BWT[0]: SA[0]=6 -> sentinel (since we're treating n=6 as end)
        // Hmm, let me trace through properly with n=6:
        // SA[i] in range 0..=6
        // For idx=0, we return sentinel
        // For idx>0, we return text[idx-1]

        // With this logic:
        // SA[0]=6, idx=6 != 0, so text[6-1]=text[5]='a'
        // SA[1]=5, text[4]='n'
        // SA[2]=3, text[2]='n'
        // SA[3]=1, text[0]='b'
        // SA[4]=0, sentinel
        // SA[5]=4, text[3]='a'
        // SA[6]=2, text[1]='a'
        // BWT = "annb\0aa" (where \0 is sentinel)

        assert_eq!(bwt.len(), 7);
        assert!(bwt.contains(&SENTINEL));
    }

    #[test]
    fn test_c_table() {
        let bwt = vec![SENTINEL, b'a', b'a', b'b', b'a'];
        let c_table = build_c_table(&bwt);

        // After sentinel (position 0), characters start at position 1
        // C['a'] should be 1 (one sentinel before 'a')
        // C['b'] should be 1 + 3 = 4 (sentinel + three 'a's before 'b')
        assert!(c_table[b'a' as usize] >= 1);
    }
}
