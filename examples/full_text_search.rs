//! FM-Index Full-Text Search Example
//!
//! Demonstrates building an index and searching in O(pattern_length).
//!
//! ```bash
//! cargo run --example full_text_search
//! ```

use alice_search::AliceIndex;

fn main() {
    println!("=== ALICE-Search FM-Index Demo ===\n");

    let text = b"the quick brown fox jumps over the lazy dog. \
                 the fox was quick and the dog was lazy. \
                 a quick brown dog outfoxed a lazy fox.";

    println!("Text ({} bytes):", text.len());
    println!("  \"{}\"", std::str::from_utf8(text).unwrap());

    // Build FM-Index (SA sample step = 4)
    let index = AliceIndex::build(text, 4);

    println!("\n--- Search Results ---\n");

    let queries = ["fox", "the", "quick", "lazy", "cat", "brown fox"];

    for query in &queries {
        let count = index.count(query.as_bytes());
        let positions = index.locate_all(query.as_bytes());

        if count > 0 {
            println!("  \"{}\" -> {} occurrences at positions {:?}", query, count, positions);
        } else {
            println!("  \"{}\" -> not found", query);
        }
    }

    // Demonstrate O(m) complexity (independent of text size)
    println!("\n--- Complexity Demo ---\n");
    println!("  count(\"fox\")        = {} (O(3) operations)", index.count(b"fox"));
    println!("  count(\"quick brown\") = {} (O(11) operations)", index.count(b"quick brown"));
    println!("  contains(\"cat\")     = {} (O(3) operations)", index.contains(b"cat"));
    println!("\n  Query time is O(pattern_length), independent of corpus size.");

    // Iterator-based locate (zero allocation)
    println!("\n--- Zero-Allocation Iterator ---\n");
    print!("  Positions of \"the\": ");
    for pos in index.locate(b"the") {
        print!("{} ", pos);
    }
    println!();
}
