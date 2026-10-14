# Changelog

All notable changes to ALICE-Search will be documented in this file.

## [Unreleased]

### Changed
- feature `db` が `alice-db/fs` を有効にする (`AliceDB::open` は file-backed storage の `fs` feature が必要で、default-features を切った依存では `--features db` が compile できなかった) 兄弟 ALICE-DB 0.5.0 で `cargo check --all-targets --features db` を確認
- **License: `AGPL-3.0` → `AGPL-3.0 OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` rename / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ・ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`

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
