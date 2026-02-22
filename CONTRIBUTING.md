# Contributing to ALICE-Search

## Build

```bash
cargo build
cargo build --no-default-features --features alloc   # no_std check
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **O(M) count**: pattern search is independent of corpus size — only pattern length matters.
- **SA-IS**: suffix array construction in O(N) time and O(N) space.
- **Interleaved bit-vector**: header + body blocks are co-located for L1 cache locality.
- **Zero-allocation locate**: results are enumerated via an iterator — no `Vec` allocation.
- **`no_std` + `alloc`**: core index structures require only `alloc` (no `std`).
