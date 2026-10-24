# Changelog

All notable changes to ALICE-Search will be documented in this file.

## [Unreleased]

### Changed
- Licence: `AGPL-3.0 OR LicenseRef-Commercial` → `LicenseRef-Proprietary`
- `db` feature: `alice-db` を sibling の path 依存から crates.io の `0.5` に変更し、`alice-db/fs` を有効にする (`AliceDB::open` は alice-db 0.5 で `fs` feature 配下) 以前の設定では `--features db` が compile できなかった
- License: `AGPL-3.0` → `AGPL-3.0 OR LicenseRef-Commercial` (2026-09-27)

## [0.1.0] - 2026-02-23

### Added
- `AliceIndex` — FM-Index based full-text search (build, count, contains, locate)
- `bitvec` — interleaved bit-vector with rank/select (L1 cache-aligned layout)
- `wavelet` — wavelet matrix for alphabet-general rank queries (double-buffered build)
- `bwt` — Burrows-Wheeler Transform via SA-IS (suffix array induced sorting)
- Zero-allocation `locate` iterator
- Bridge modules: `text_bridge`, `analytics_bridge`, `db_bridge`, `cache_bridge`
- `no_std` compatible core (with `alloc`)
- 47 tests (44 unit + 3 doc-test)
