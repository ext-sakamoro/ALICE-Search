# Changelog

All notable changes to ALICE-Search will be documented in this file.

## [Unreleased]

### Changed

- CI: `scripts/feature_check.py` が optional feature ごとの clippy に加えて `cargo test --lib --features <名前>` を走らせ、最後に全 feature を同時に有効にした clippy (`--all-targets`) と test を走らせる 一緒に有効にできない feature は `.github/feature-check-apart.txt` に理由付きで書く (self-test 16 通り)

### Fixed
- std なし (`default-features = false`) で build できなかった: `src/bwt.rs` が `vec!` を import せず、`std::cmp::Ordering` を使っていた (`alloc::vec` と `core::cmp` に変更) CI に std の無い target (`thumbv7em-none-eabihf`) での clippy を追加し、`rust-toolchain.toml` に target を足した

### Added
- `text` feature: `CompressedSearchIndex::search(query, context_radius)` は各出現の位置と前後 `context_radius` byte の範囲 (`SearchHit`、text の長さで切る) を返し、`search_count(query)` は文字列の出現数を返す (ALICE-Text の `search` feature から移した) 試験: 出現の位置と範囲を `str::match_indices` から作った値と比べる、空の text、圧縮した入力
- CI: `scripts/feature_check.py` が optional feature を 1 つずつ `cargo clippy --lib --features <名前> -- -D warnings` で build する (Cargo.toml の `[features]` から読む、その feature の code だけが出す警告 (捨てた `Result` など) も失敗にする、build した数が 0 なら失敗、self-test 11 通り) CI でまだ空の stub に対して build する feature は理由付きで `.github/feature-check-pending.txt` に書き、`PENDING` と出す 手元の preflight は実物の sibling で全 feature を build する (`scripts/preflight.local.sh`)

### Changed
- 依存: alice-text / alice-analytics / alice-cache を固定 rev でなく、alice-db を crates.io の 0.5 でなく、各 repository の main branch と版で引く (alice-db は 0.6) 生成される lock は alice-det-math を git main の 1 つだけ持ち、crates.io の alice-* は 0 (これまでは 0.5.1 と 0.5.2 の 2 世代)
- 依存: `alice-text` (feature `text`) を ALICE-Text の commit `ab5d154` (2.0.0) の git で取る path の sibling が無くなったので、CI で空の stub を作る `.github/make-stubs.py` と その step、`text` を保留にしていた `.github/feature-check-pending.txt`、手元だけで全 feature を build していた `scripts/preflight.local.sh` を削除した CI の feature ごとの build は `text` も実物で build する
- `analytics` feature: `alice_analytics::prelude` (ALICE-Analytics に無い) でなく `alice_analytics::sketch` から `HyperLogLog` / `DDSketch` / `CountMinSketch` を引く 以前は `--features analytics` が compile できなかった
- 依存: `alice-analytics` を ALICE-Analytics の commit `f3a58e3` (0.7.0)、`alice-cache` を ALICE-Cache の commit `48ffb79` (0.2.0) の git で取る (sibling の path でなくなり、CI でも実物を build する)
- CI: `--features db` だけを build していた step は feature ごとの build に含めた
- Licence: `AGPL-3.0 OR LicenseRef-Commercial` → `LicenseRef-Proprietary`
- `db` feature: `alice-db` を sibling の path 依存から crates.io の `0.5` に変更し、`alice-db/fs` を有効にする (`AliceDB::open` は alice-db 0.5 で `fs` feature 配下) 以前の設定では `--features db` が compile できなかった
- License: `AGPL-3.0` → `AGPL-3.0 OR LicenseRef-Commercial` (2026-09-27)
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る

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
