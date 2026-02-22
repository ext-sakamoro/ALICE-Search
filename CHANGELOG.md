# Changelog

All notable changes to ALICE-Search will be documented in this file.

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
