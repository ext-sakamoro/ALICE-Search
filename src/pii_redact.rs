//! `pii_redact` — Personally Identifiable Information detection and redaction.
//!
//! Complements the `FM-Index` full-text search with a PII scrubber suitable
//! for indexing pipelines that must strip regulated identifiers before
//! documents land in the index.
//!
//! Supported patterns (English-language conventions unless noted):
//!
//! - **Email addresses** — `local@domain.tld` with a permissive charset.
//! - **IPv4 addresses** — four dotted decimal octets.
//! - **Credit card numbers** — 13-19 digits, verified with the Luhn
//!   checksum (`ISO/IEC 7812-1`).
//! - **US SSN** — `NNN-NN-NNNN`.
//! - **Japanese postal code** — `NNN-NNNN` (7 digits with hyphen).
//! - **Japanese phone number** — `0N-NNNN-NNNN` or `0NN-NNN-NNNN`.
//! - **Japanese My Number** — 12 consecutive digits with a valid check
//!   digit (`地方公共団体情報システム機構` §10).
//!
//! # Regulatory alignment
//!
//! - **`GDPR` Art. 4(1) + Art. 9** — personal data and special-category
//!   data must be minimized before further processing.
//! - **`PCI-DSS` v4.0 §3.4** — Primary Account Number storage requires
//!   masking; a `redact` pass is one of the accepted mitigations.
//! - **改正個人情報保護法 §2 (My Number Act §14)** — 特定個人情報の限定的
//!   利用と削除 (redaction 経路の実装が推奨される).
//! - **`HIPAA` Safe Harbor** — 18 identifiers must be removed for
//!   de-identification.

#![allow(
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::missing_const_for_fn
)]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// PiiKind
// ---------------------------------------------------------------------------

/// Category of PII a match belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiKind {
    /// Email address.
    Email,
    /// IPv4 address in dotted decimal notation.
    IpV4,
    /// Credit card number (validated with the Luhn checksum).
    CreditCard,
    /// US Social Security Number in `NNN-NN-NNNN` form.
    UsSsn,
    /// Japanese postal code `NNN-NNNN`.
    JpPostalCode,
    /// Japanese phone number.
    JpPhone,
    /// Japanese My Number (12 digits with valid check).
    JpMyNumber,
}

impl PiiKind {
    /// Placeholder tag used by [`redact_placeholder`].
    #[must_use]
    pub const fn placeholder(&self) -> &'static str {
        match self {
            Self::Email => "<EMAIL>",
            Self::IpV4 => "<IPV4>",
            Self::CreditCard => "<CREDIT_CARD>",
            Self::UsSsn => "<US_SSN>",
            Self::JpPostalCode => "<JP_POSTAL>",
            Self::JpPhone => "<JP_PHONE>",
            Self::JpMyNumber => "<JP_MYNUMBER>",
        }
    }
}

// ---------------------------------------------------------------------------
// PiiMatch
// ---------------------------------------------------------------------------

/// One PII occurrence in the source text.
///
/// `start` and `end` are byte offsets into the original `&str`. `matched`
/// is the substring for convenience.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiiMatch {
    /// PII category.
    pub kind: PiiKind,
    /// Byte offset (inclusive) of the match start.
    pub start: usize,
    /// Byte offset (exclusive) of the match end.
    pub end: usize,
    /// The exact substring that matched.
    pub matched: String,
}

// ---------------------------------------------------------------------------
// detect_pii
// ---------------------------------------------------------------------------

/// Scan `text` and return all PII matches, sorted by `start` offset.
///
/// Matches are non-overlapping: when two patterns compete for the same
/// bytes, the earliest-starting match wins; ties are broken by longest.
#[must_use]
pub fn detect_pii(text: &str) -> Vec<PiiMatch> {
    let mut hits: Vec<PiiMatch> = Vec::new();
    scan_email(text, &mut hits);
    scan_ipv4(text, &mut hits);
    scan_credit_card(text, &mut hits);
    scan_us_ssn(text, &mut hits);
    scan_jp_postal(text, &mut hits);
    scan_jp_phone(text, &mut hits);
    scan_jp_mynumber(text, &mut hits);
    resolve_overlaps(hits)
}

fn resolve_overlaps(mut hits: Vec<PiiMatch>) -> Vec<PiiMatch> {
    hits.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
    let mut out: Vec<PiiMatch> = Vec::with_capacity(hits.len());
    let mut cursor = 0usize;
    for m in hits {
        if m.start >= cursor {
            cursor = m.end;
            out.push(m);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

/// Return a copy of `text` with every PII byte replaced by `mask`.
///
/// Only characters that fall inside a [`PiiMatch`] span are replaced; the
/// surrounding text is preserved. The mask character must be `ASCII`.
#[must_use]
pub fn redact(text: &str, mask: char) -> String {
    let matches = detect_pii(text);
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for m in matches {
        out.push_str(&text[cursor..m.start]);
        for _ in 0..(m.end - m.start) {
            out.push(mask);
        }
        cursor = m.end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Return a copy of `text` with every PII span replaced by
/// [`PiiKind::placeholder`].
#[must_use]
pub fn redact_placeholder(text: &str) -> String {
    let matches = detect_pii(text);
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for m in matches {
        out.push_str(&text[cursor..m.start]);
        out.push_str(m.kind.placeholder());
        cursor = m.end;
    }
    out.push_str(&text[cursor..]);
    out
}

// ---------------------------------------------------------------------------
// Individual scanners
// ---------------------------------------------------------------------------

fn scan_email(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' {
            let start = local_start(bytes, i);
            let end = domain_end(bytes, i);
            if start < i && end > i + 1 {
                let matched = &text[start..end];
                if is_valid_email(matched) {
                    out.push(PiiMatch {
                        kind: PiiKind::Email,
                        start,
                        end,
                        matched: matched.to_string(),
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
}

fn is_email_local(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'+' | b'-' | b'%')
}

fn is_email_domain(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-')
}

fn local_start(bytes: &[u8], at: usize) -> usize {
    let mut s = at;
    while s > 0 && is_email_local(bytes[s - 1]) {
        s -= 1;
    }
    s
}

fn domain_end(bytes: &[u8], at: usize) -> usize {
    let mut e = at + 1;
    while e < bytes.len() && is_email_domain(bytes[e]) {
        e += 1;
    }
    e
}

fn is_valid_email(candidate: &str) -> bool {
    let Some(at) = candidate.find('@') else {
        return false;
    };
    let (local, rest) = candidate.split_at(at);
    let domain = &rest[1..];
    if local.is_empty() || domain.is_empty() {
        return false;
    }
    if !domain.contains('.') {
        return false;
    }
    let last = domain.rsplit('.').next().unwrap_or("");
    if last.len() < 2 {
        return false;
    }
    last.bytes().all(|b| b.is_ascii_alphabetic())
}

fn scan_ipv4(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit()
            && (i == 0 || !bytes[i - 1].is_ascii_digit() && bytes[i - 1] != b'.')
        {
            if let Some(end) = try_ipv4(bytes, i) {
                out.push(PiiMatch {
                    kind: PiiKind::IpV4,
                    start: i,
                    end,
                    matched: text[i..end].to_string(),
                });
                i = end;
                continue;
            }
        }
        i += 1;
    }
}

fn try_ipv4(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    for octet_index in 0..4 {
        let octet_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let len = i - octet_start;
        if !(1..=3).contains(&len) {
            return None;
        }
        let mut val = 0u32;
        for j in octet_start..i {
            val = val * 10 + u32::from(bytes[j] - b'0');
        }
        if val > 255 {
            return None;
        }
        if octet_index < 3 {
            if i >= bytes.len() || bytes[i] != b'.' {
                return None;
            }
            i += 1;
        }
    }
    if i < bytes.len() && bytes[i].is_ascii_digit() {
        return None;
    }
    Some(i)
}

fn scan_credit_card(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() && (i == 0 || !bytes[i - 1].is_ascii_digit()) {
            if let Some((digit_count, end)) = try_pan(bytes, i) {
                if (13..=19).contains(&digit_count) && luhn(&text[i..end]) {
                    out.push(PiiMatch {
                        kind: PiiKind::CreditCard,
                        start: i,
                        end,
                        matched: text[i..end].to_string(),
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
}

const fn is_pan_body(b: u8) -> bool {
    b.is_ascii_digit() || b == b' ' || b == b'-'
}

fn try_pan(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut i = start;
    let mut digits = 0usize;
    while i < bytes.len() && is_pan_body(bytes[i]) {
        if bytes[i].is_ascii_digit() {
            digits += 1;
        }
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    while i > start && !bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    Some((digits, i))
}

fn luhn(text: &str) -> bool {
    let digits: Vec<u32> = text
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| u32::from(b - b'0'))
        .collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    for (i, d) in digits.iter().rev().enumerate() {
        if i % 2 == 1 {
            let doubled = d * 2;
            sum += if doubled > 9 { doubled - 9 } else { doubled };
        } else {
            sum += *d;
        }
    }
    sum % 10 == 0
}

fn scan_us_ssn(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n < 11 {
        return;
    }
    let mut i = 0;
    while i + 11 <= n {
        if is_ascii_digit_slice(&bytes[i..i + 3])
            && bytes[i + 3] == b'-'
            && is_ascii_digit_slice(&bytes[i + 4..i + 6])
            && bytes[i + 6] == b'-'
            && is_ascii_digit_slice(&bytes[i + 7..i + 11])
            && (i == 0 || !bytes[i - 1].is_ascii_digit())
            && (i + 11 == n || !bytes[i + 11].is_ascii_digit())
        {
            out.push(PiiMatch {
                kind: PiiKind::UsSsn,
                start: i,
                end: i + 11,
                matched: text[i..i + 11].to_string(),
            });
            i += 11;
            continue;
        }
        i += 1;
    }
}

fn is_ascii_digit_slice(b: &[u8]) -> bool {
    b.iter().all(u8::is_ascii_digit)
}

fn scan_jp_postal(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n < 8 {
        return;
    }
    let mut i = 0;
    while i + 8 <= n {
        if is_ascii_digit_slice(&bytes[i..i + 3])
            && bytes[i + 3] == b'-'
            && is_ascii_digit_slice(&bytes[i + 4..i + 8])
            && (i == 0 || !bytes[i - 1].is_ascii_digit())
            && (i + 8 == n || !bytes[i + 8].is_ascii_digit())
        {
            out.push(PiiMatch {
                kind: PiiKind::JpPostalCode,
                start: i,
                end: i + 8,
                matched: text[i..i + 8].to_string(),
            });
            i += 8;
            continue;
        }
        i += 1;
    }
}

fn scan_jp_phone(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if bytes[i] == b'0' && (i == 0 || !bytes[i - 1].is_ascii_digit()) {
            if let Some(end) = try_jp_phone(bytes, i) {
                out.push(PiiMatch {
                    kind: PiiKind::JpPhone,
                    start: i,
                    end,
                    matched: text[i..end].to_string(),
                });
                i = end;
                continue;
            }
        }
        i += 1;
    }
}

/// Match `0NN[N]-NNN[N]-NNNN` shapes, e.g. `03-1234-5678`,
/// `090-1234-5678`, `0120-123-456`.
fn try_jp_phone(bytes: &[u8], start: usize) -> Option<usize> {
    let n = bytes.len();
    let mut i = start;
    // Area code: leading 0 + 1..=3 more digits.
    if i >= n || bytes[i] != b'0' {
        return None;
    }
    i += 1;
    let mut area_extra = 0usize;
    while area_extra < 3 && i < n && bytes[i].is_ascii_digit() {
        i += 1;
        area_extra += 1;
    }
    if area_extra == 0 {
        return None;
    }
    if i >= n || bytes[i] != b'-' {
        return None;
    }
    i += 1;
    // Exchange: 3..=4 digits.
    let exchange_start = i;
    while i < n && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let exchange_len = i - exchange_start;
    if !(3..=4).contains(&exchange_len) {
        return None;
    }
    if i >= n || bytes[i] != b'-' {
        return None;
    }
    i += 1;
    // Subscriber: exactly 4 digits.
    let sub_start = i;
    while i < n && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let sub_len = i - sub_start;
    if sub_len != 4 {
        return None;
    }
    if i < n && bytes[i].is_ascii_digit() {
        return None;
    }
    Some(i)
}

fn scan_jp_mynumber(text: &str, out: &mut Vec<PiiMatch>) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n < 12 {
        return;
    }
    let mut i = 0;
    while i + 12 <= n {
        if is_ascii_digit_slice(&bytes[i..i + 12])
            && (i == 0 || !bytes[i - 1].is_ascii_digit())
            && (i + 12 == n || !bytes[i + 12].is_ascii_digit())
            && jp_mynumber_valid(&bytes[i..i + 12])
        {
            out.push(PiiMatch {
                kind: PiiKind::JpMyNumber,
                start: i,
                end: i + 12,
                matched: text[i..i + 12].to_string(),
            });
            i += 12;
            continue;
        }
        i += 1;
    }
}

/// My Number check digit (`地方公共団体情報システム機構` §10):
///
/// - Sum `d1..d11` weighted by `[6,5,4,3,2,7,6,5,4,3,2]`.
/// - Take `remainder = sum mod 11`.
/// - If `remainder <= 1`, check digit is 0; otherwise `11 - remainder`.
fn jp_mynumber_valid(digits: &[u8]) -> bool {
    if digits.len() != 12 {
        return false;
    }
    let weights: [u32; 11] = [6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2];
    let mut sum: u32 = 0;
    for i in 0..11 {
        sum += u32::from(digits[i] - b'0') * weights[i];
    }
    let remainder = sum % 11;
    let expected = if remainder <= 1 { 0 } else { 11 - remainder };
    u32::from(digits[11] - b'0') == expected
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_stable() {
        assert_eq!(PiiKind::Email.placeholder(), "<EMAIL>");
        assert_eq!(PiiKind::IpV4.placeholder(), "<IPV4>");
        assert_eq!(PiiKind::CreditCard.placeholder(), "<CREDIT_CARD>");
        assert_eq!(PiiKind::UsSsn.placeholder(), "<US_SSN>");
        assert_eq!(PiiKind::JpPostalCode.placeholder(), "<JP_POSTAL>");
        assert_eq!(PiiKind::JpPhone.placeholder(), "<JP_PHONE>");
        assert_eq!(PiiKind::JpMyNumber.placeholder(), "<JP_MYNUMBER>");
    }

    #[test]
    fn detects_simple_email() {
        let hits = detect_pii("contact alice@example.com now");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, PiiKind::Email);
        assert_eq!(hits[0].matched, "alice@example.com");
    }

    #[test]
    fn ignores_bare_at_sign() {
        assert!(detect_pii("no email here @").is_empty());
    }

    #[test]
    fn detects_ipv4() {
        let hits = detect_pii("server 192.168.1.10 down");
        assert!(hits
            .iter()
            .any(|h| h.kind == PiiKind::IpV4 && h.matched == "192.168.1.10"));
    }

    #[test]
    fn rejects_ipv4_out_of_range() {
        // 999 > 255, so this should not be detected as an IPv4.
        let hits = detect_pii("999.999.999.999");
        assert!(hits.iter().all(|h| h.kind != PiiKind::IpV4));
    }

    #[test]
    fn detects_luhn_valid_credit_card() {
        // 4242 4242 4242 4242 is a Stripe test PAN with a valid Luhn checksum.
        let hits = detect_pii("PAN 4242 4242 4242 4242 x");
        assert!(hits.iter().any(|h| h.kind == PiiKind::CreditCard));
    }

    #[test]
    fn rejects_luhn_invalid_credit_card() {
        // 4242 4242 4242 4243 fails the Luhn checksum.
        let hits = detect_pii("PAN 4242 4242 4242 4243 x");
        assert!(hits.iter().all(|h| h.kind != PiiKind::CreditCard));
    }

    #[test]
    fn detects_us_ssn() {
        let hits = detect_pii("SSN 123-45-6789 filed.");
        assert!(hits
            .iter()
            .any(|h| h.kind == PiiKind::UsSsn && h.matched == "123-45-6789"));
    }

    #[test]
    fn detects_jp_postal() {
        let hits = detect_pii("〒 100-0001 東京都千代田区");
        assert!(hits
            .iter()
            .any(|h| h.kind == PiiKind::JpPostalCode && h.matched == "100-0001"));
    }

    #[test]
    fn detects_jp_phone() {
        let hits = detect_pii("TEL 03-1234-5678 内線 100");
        assert!(hits
            .iter()
            .any(|h| h.kind == PiiKind::JpPhone && h.matched == "03-1234-5678"));
    }

    #[test]
    fn detects_valid_jp_mynumber() {
        // Sum weighted: pick a number whose check digit matches.
        // Weights: 6 5 4 3 2 7 6 5 4 3 2
        // Digits d1..d11 = 1 2 3 4 5 6 7 8 9 0 1
        // 6+10+12+12+10+42+42+40+36+0+2 = 212 → 212 % 11 = 3 → check = 8
        let hits = detect_pii("番号 123456789018 です");
        assert!(hits
            .iter()
            .any(|h| h.kind == PiiKind::JpMyNumber && h.matched == "123456789018"));
    }

    #[test]
    fn rejects_invalid_jp_mynumber() {
        let hits = detect_pii("番号 123456789017 です");
        assert!(hits.iter().all(|h| h.kind != PiiKind::JpMyNumber));
    }

    #[test]
    fn redact_replaces_pii_bytes_with_mask() {
        let redacted = redact("Contact alice@example.com!", 'X');
        assert_eq!(redacted, "Contact XXXXXXXXXXXXXXXXX!");
    }

    #[test]
    fn redact_placeholder_swaps_to_tag() {
        let redacted = redact_placeholder("Email alice@example.com then call 03-1234-5678.");
        assert_eq!(redacted, "Email <EMAIL> then call <JP_PHONE>.");
    }

    #[test]
    fn redact_preserves_non_pii_text() {
        let redacted = redact("no pii here", '*');
        assert_eq!(redacted, "no pii here");
    }

    #[test]
    fn detect_returns_matches_in_order() {
        let hits = detect_pii("call 03-1234-5678 then email alice@example.com");
        assert!(hits.len() >= 2);
        for pair in hits.windows(2) {
            assert!(pair[0].start <= pair[1].start);
        }
    }

    #[test]
    fn overlapping_matches_resolve_by_earliest() {
        // The IPv4 pattern must not eat digits that also look like a PAN.
        let hits = detect_pii("10.0.0.1");
        let ipv4_count = hits.iter().filter(|h| h.kind == PiiKind::IpV4).count();
        assert_eq!(ipv4_count, 1);
    }

    #[test]
    fn empty_input_returns_no_matches() {
        assert!(detect_pii("").is_empty());
    }
}
