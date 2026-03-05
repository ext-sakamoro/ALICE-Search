// ALICE-Search — UE5 C++ Bindings
// License: AGPL-3.0
// Author: Moroya Sakamoto
//
// 10 extern "C" declarations + RAII wrapper class

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

// ── extern "C" declarations ─────────────────────────────────────────

extern "C" {

void*    alice_index_build(const uint8_t* text, uint32_t text_len, uint32_t sample_step);
uint32_t alice_index_count(const void* index, const uint8_t* pattern, uint32_t pattern_len);
uint32_t* alice_index_locate(const void* index, const uint8_t* pattern,
             uint32_t pattern_len, uint32_t* out_len);
void     alice_index_locate_free(uint32_t* ptr, uint32_t len);
uint8_t  alice_index_contains(const void* index, const uint8_t* pattern, uint32_t pattern_len);
uint32_t alice_index_size_bytes(const void* index);
uint32_t alice_index_text_len(const void* index);
double   alice_index_compression_ratio(const void* index);
uint32_t alice_index_sample_step(const void* index);
void     alice_index_destroy(void* index);

} // extern "C"

// ── RAII wrapper ────────────────────────────────────────────────────

class AliceSearchIndex {
    void* ptr_;
public:
    AliceSearchIndex(const uint8_t* text, uint32_t len, uint32_t sample_step = 4)
        : ptr_(alice_index_build(text, len, sample_step)) {}

    ~AliceSearchIndex() { if (ptr_) alice_index_destroy(ptr_); }

    AliceSearchIndex(const AliceSearchIndex&) = delete;
    AliceSearchIndex& operator=(const AliceSearchIndex&) = delete;
    AliceSearchIndex(AliceSearchIndex&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)) {}
    AliceSearchIndex& operator=(AliceSearchIndex&& o) noexcept {
        if (this != &o) { if (ptr_) alice_index_destroy(ptr_); ptr_ = std::exchange(o.ptr_, nullptr); }
        return *this;
    }

    uint32_t Count(const uint8_t* pattern, uint32_t len) const {
        return alice_index_count(ptr_, pattern, len);
    }

    std::vector<uint32_t> Locate(const uint8_t* pattern, uint32_t len) const {
        uint32_t out_len = 0;
        auto* raw = alice_index_locate(ptr_, pattern, len, &out_len);
        if (!raw || out_len == 0) return {};
        std::vector<uint32_t> result(raw, raw + out_len);
        alice_index_locate_free(raw, out_len);
        return result;
    }

    bool Contains(const uint8_t* pattern, uint32_t len) const {
        return alice_index_contains(ptr_, pattern, len) != 0;
    }

    uint32_t SizeBytes() const { return alice_index_size_bytes(ptr_); }
    uint32_t TextLen() const { return alice_index_text_len(ptr_); }
    double CompressionRatio() const { return alice_index_compression_ratio(ptr_); }
    uint32_t SampleStep() const { return alice_index_sample_step(ptr_); }
};
