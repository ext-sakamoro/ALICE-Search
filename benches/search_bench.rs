use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use alice_search::AliceIndex;

fn generate_text(size: usize) -> Vec<u8> {
    let words = [
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
        "alice ", "bob ", "server ", "request ", "response ", "error ", "data ",
        "cache ", "index ", "search ", "query ", "result ",
    ];
    let mut text = Vec::with_capacity(size);
    let mut i = 0;
    while text.len() < size {
        let word = words[i % words.len()].as_bytes();
        text.extend_from_slice(word);
        i += 1;
    }
    text.truncate(size);
    text
}

fn bench_build_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_index");

    for size in [1_000, 10_000, 50_000] {
        let text = generate_text(size);
        group.bench_with_input(
            BenchmarkId::new("bytes", size),
            &text,
            |b, text| {
                b.iter(|| AliceIndex::build(black_box(text), 4))
            },
        );
    }
    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let text = generate_text(100_000);
    let index = AliceIndex::build(&text, 4);

    let mut group = c.benchmark_group("count");

    for pattern in ["fox", "the quick", "server request response"] {
        group.bench_with_input(
            BenchmarkId::new("pattern", pattern),
            pattern.as_bytes(),
            |b, pat| {
                b.iter(|| index.count(black_box(pat)))
            },
        );
    }
    group.finish();
}

fn bench_contains(c: &mut Criterion) {
    let text = generate_text(100_000);
    let index = AliceIndex::build(&text, 4);

    c.bench_function("contains_hit", |b| {
        b.iter(|| index.contains(black_box(b"fox")))
    });

    c.bench_function("contains_miss", |b| {
        b.iter(|| index.contains(black_box(b"zzzzz")))
    });
}

fn bench_locate(c: &mut Criterion) {
    let text = generate_text(100_000);
    let index = AliceIndex::build(&text, 4);

    c.bench_function("locate_all_fox", |b| {
        b.iter(|| {
            let positions = index.locate_all(black_box(b"fox"));
            black_box(positions.len())
        })
    });

    c.bench_function("locate_iter_first_10", |b| {
        b.iter(|| {
            let count = index.locate(black_box(b"the")).take(10).count();
            black_box(count)
        })
    });
}

criterion_group!(
    benches,
    bench_build_index,
    bench_count,
    bench_contains,
    bench_locate,
);
criterion_main!(benches);
