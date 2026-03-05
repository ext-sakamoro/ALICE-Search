//! C FFI for ALICE-Search
//!
//! Provides 10 `extern "C"` functions for Unity / UE5 / native integration.
//!
//! License: AGPL-3.0
//! Author: Moroya Sakamoto

use std::ptr;
use std::slice;

use crate::search::AliceIndex;

// ── AliceIndex (10) ─────────────────────────────────────────────────

/// テキストからインデックスを構築する。
///
/// # Safety
///
/// `text_ptr`は`text_len`バイトの有効な読み取り可能メモリであること。
/// 戻り値は`alice_index_destroy`で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_build(
    text_ptr: *const u8,
    text_len: u32,
    sample_step: u32,
) -> *mut AliceIndex {
    if text_ptr.is_null() || text_len == 0 {
        return ptr::null_mut();
    }
    let text = slice::from_raw_parts(text_ptr, text_len as usize);
    let index = AliceIndex::build(text, sample_step as usize);
    Box::into_raw(Box::new(index))
}

/// パターンの出現回数を返す。
///
/// # Safety
///
/// `index`は`alice_index_build`で取得した有効なポインタであること。
/// `pattern_ptr`は`pattern_len`バイトの有効な読み取り可能メモリであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_count(
    index: *const AliceIndex,
    pattern_ptr: *const u8,
    pattern_len: u32,
) -> u32 {
    if index.is_null() {
        return 0;
    }
    let pattern = if pattern_ptr.is_null() || pattern_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(pattern_ptr, pattern_len as usize)
    };
    (*index).count(pattern) as u32
}

/// パターンの全出現位置を返す。
///
/// # Safety
///
/// `index`は有効なポインタであること。`pattern_ptr`は`pattern_len`バイトの有効なメモリであること。
/// `out_len`は有効なポインタであること。
/// 戻り値は`alice_index_locate_free`で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_locate(
    index: *const AliceIndex,
    pattern_ptr: *const u8,
    pattern_len: u32,
    out_len: *mut u32,
) -> *mut u32 {
    if index.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }
    let pattern = if pattern_ptr.is_null() || pattern_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(pattern_ptr, pattern_len as usize)
    };
    let positions: Vec<u32> = (*index)
        .locate_all(pattern)
        .iter()
        .map(|&p| p as u32)
        .collect();
    let len = positions.len();
    *out_len = len as u32;

    if len == 0 {
        return ptr::null_mut();
    }

    let mut boxed = positions.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// `alice_index_locate`で返された配列を解放する。
///
/// # Safety
///
/// `ptr`は`alice_index_locate`で取得したポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_locate_free(ptr: *mut u32, len: u32) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len as usize, len as usize));
    }
}

/// パターンが存在するか確認する。存在=1, 不在=0。
///
/// # Safety
///
/// `index`は有効なポインタであること。`pattern_ptr`は`pattern_len`バイトの有効なメモリであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_contains(
    index: *const AliceIndex,
    pattern_ptr: *const u8,
    pattern_len: u32,
) -> u8 {
    if index.is_null() {
        return 0;
    }
    let pattern = if pattern_ptr.is_null() || pattern_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(pattern_ptr, pattern_len as usize)
    };
    u8::from((*index).contains(pattern))
}

/// インデックスの概算バイトサイズを返す。
///
/// # Safety
///
/// `index`は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_size_bytes(index: *const AliceIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).size_bytes() as u32
}

/// 元テキストの長さを返す。
///
/// # Safety
///
/// `index`は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_text_len(index: *const AliceIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).text_len() as u32
}

/// 圧縮率を返す。
///
/// # Safety
///
/// `index`は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_compression_ratio(index: *const AliceIndex) -> f64 {
    if index.is_null() {
        return 0.0;
    }
    (*index).compression_ratio()
}

/// SAサンプリングステップを返す。
///
/// # Safety
///
/// `index`は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_index_sample_step(index: *const AliceIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).sample_step() as u32
}

/// AliceIndexを解放する。
///
/// # Safety
///
/// `index`は`alice_index_build`で取得したポインタであること。
/// 解放後にこのポインタを使用してはならない。
#[no_mangle]
pub unsafe extern "C" fn alice_index_destroy(index: *mut AliceIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

// ── テスト ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_lifecycle() {
        unsafe {
            let text = b"abracadabra";
            let index = alice_index_build(text.as_ptr(), text.len() as u32, 4);
            assert!(!index.is_null());

            assert_eq!(alice_index_count(index, b"abra".as_ptr(), 4), 2);
            assert_eq!(alice_index_count(index, b"xyz".as_ptr(), 3), 0);
            assert_eq!(alice_index_contains(index, b"cadabra".as_ptr(), 7), 1);
            assert_eq!(alice_index_contains(index, b"xyz".as_ptr(), 3), 0);

            assert_eq!(alice_index_text_len(index), 11);
            assert_eq!(alice_index_sample_step(index), 4);
            assert!(alice_index_size_bytes(index) > 0);
            assert!(alice_index_compression_ratio(index) > 0.0);

            alice_index_destroy(index);
        }
    }

    #[test]
    fn test_locate_and_free() {
        unsafe {
            let text = b"abracadabra";
            let index = alice_index_build(text.as_ptr(), text.len() as u32, 1);

            let mut len: u32 = 0;
            let positions = alice_index_locate(index, b"abra".as_ptr(), 4, &mut len);
            assert_eq!(len, 2);
            assert!(!positions.is_null());

            let slice = slice::from_raw_parts(positions, len as usize);
            let mut sorted: Vec<u32> = slice.to_vec();
            sorted.sort();
            assert_eq!(sorted, vec![0, 7]);

            alice_index_locate_free(positions, len);
            alice_index_destroy(index);
        }
    }

    #[test]
    fn test_locate_no_match() {
        unsafe {
            let text = b"hello";
            let index = alice_index_build(text.as_ptr(), text.len() as u32, 4);

            let mut len: u32 = 0;
            let positions = alice_index_locate(index, b"xyz".as_ptr(), 3, &mut len);
            assert_eq!(len, 0);
            assert!(positions.is_null());

            alice_index_destroy(index);
        }
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            assert!(alice_index_build(ptr::null(), 0, 4).is_null());
            assert_eq!(alice_index_count(ptr::null(), b"a".as_ptr(), 1), 0);
            assert_eq!(alice_index_contains(ptr::null(), b"a".as_ptr(), 1), 0);
            assert_eq!(alice_index_size_bytes(ptr::null()), 0);
            assert_eq!(alice_index_text_len(ptr::null()), 0);
            assert_eq!(alice_index_sample_step(ptr::null()), 0);
            assert!(alice_index_compression_ratio(ptr::null()) == 0.0);

            alice_index_destroy(ptr::null_mut());
            alice_index_locate_free(ptr::null_mut(), 0);
        }
    }
}
