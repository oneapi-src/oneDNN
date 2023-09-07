/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_FORMAT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_FORMAT_HPP

#include <array>
#include <ostream>
#include <vector>
#include <compiler/dimensions.hpp>
#include <runtime/dispatch_key.hpp>
#include <unordered_map>
#include <util/def.hpp>
#include <util/hash_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/// Memory format kind
enum class sc_format_category {
    // any means: support block and plain
    any,
    // blocked means: only support block
    blocked,
    // vnni blocked means: specific block format used by vnni instructions
    vnni_blocked,
    // non_blocking means: plain or permuted
    non_blocking
};

/**
 * The encoded data format kind. It stores the mapping of each axis in the real
 * shape with the axis in the original shape. We interpret the 64-bit storage as
 * 16x 4-bit ints. We use the last 4-bit int (15-th) as the control block
 * [slot0],[slot1],[slot2],...[slot15]
 * The slots 0~14 (15 slots) store the original axis index of the corresponding
 * dimension. For a N-dimension format, any slots with index >=N should contain
 * a value of (0xF). For example, NCHWc =>[0,1,2,3,1,-1,-1,...]
 * The slot15 is a control block, which is initially set to be 0 and can be used
 * for other purpose in the future.
 */
struct SC_API sc_data_format_kind_t {
    static constexpr int NUM_SLOTS = 16;
    static constexpr int MAX_DIMS = NUM_SLOTS - 1; // 15
    static constexpr int BITS_PER_SLOT = sizeof(uint64_t) * 8 / NUM_SLOTS; // 4
    static constexpr int UNDEF_DIM = (1 << BITS_PER_SLOT) - 1; // 0xf
    uint64_t storage_;
    // gets the original axis of the idx-th dimemsion of the format
    constexpr int get(int idx) const {
        return 0xf & (storage_ >> (idx * BITS_PER_SLOT));
    }
    constexpr int get_control_block() const { return get(MAX_DIMS); }
    void set(int idx, int data) { storage_ = set_ith_int(storage_, idx, data); }
    constexpr sc_data_format_kind_t(uint64_t storage) : storage_(storage) {}
    constexpr sc_data_format_kind_t() : storage_(0xffffffffffffffff) {}

private:
    static constexpr uint64_t set_ith_int(uint64_t oldv, int idx, int data) {
        return (oldv & ~(uint64_t(UNDEF_DIM) << (idx * BITS_PER_SLOT)))
                | (uint64_t(data) << (idx * BITS_PER_SLOT));
    }
    template <int start, typename... Args>
    static constexpr uint64_t make_storage(uint64_t oldv, int v, Args... args) {
        return make_storage<start + 1>(set_ith_int(oldv, start, v), args...);
    }

    template <int start>
    static constexpr uint64_t make_storage(uint64_t oldv, int v) {
        return set_ith_int(oldv, start, v);
    }

public:
    template <typename... Args>
    constexpr sc_data_format_kind_t(Args... args)
        : storage_(make_storage<0>(
                set_ith_int(0xffffffffffffffff, MAX_DIMS, 0), args...)) {
        static_assert(sizeof...(args) <= MAX_DIMS,
                "At most 15 dimensions are supported");
    }

    sc_data_format_kind_t(const std::vector<int> &storage_args);

    constexpr sc_data_format_kind_t(const sc_data_format_kind_t &) = default;
    sc_data_format_kind_t &operator=(const sc_data_format_kind_t &) = default;
    constexpr bool operator==(const sc_data_format_kind_t &other) const {
        return storage_ == other.storage_;
    }

    constexpr bool operator!=(const sc_data_format_kind_t &other) const {
        return storage_ != other.storage_;
    }

    constexpr operator uint64_t() const { return storage_; }

    // gets the number of dimensions. For any, returns -1.
    int ndims() const;

    // gets the number of original dimensions. For any, returns -1.
    int norig_dims() const;

    // checks if the format is valid. If not, throws an runtime_exception
    void check() const;

    bool is_plain() const;
    bool is_blocking() const;

    bool is_channel_last() const;

    // collects the number of axies in the format. For original axis `i`,
    // `out[i]` will be the number of occurence of the axis in this format.
    // e.g. NCHWc => out=[1,2,1,1], the axis C occurs twice
    void collect_dim_count(int out[MAX_DIMS]) const;

    // collects the index of blocking with given `axis`. e.g. NCHWc and given
    // `axis=1` we get the blocking index vector {0}
    std::vector<int> collect_blocking_index(int axis) const;

    // collects the mapping from plain axis to blocking axis for each dimension.
    // e.g. NCHWc will return {{0},{1,4},{2},{3}}, MKmk will return
    // {{0,2},{1,3}}
    std::vector<std::vector<int>> collect_p2b_mapping() const;

    sc_data_format_kind_t to_plain() const;

    sc_data_format_kind_t to_channel_last() const;

    // makes an N-D plain format.
    static sc_data_format_kind_t get_plain_by_dims(size_t ndims);
    // makes a format that 2d blocking are at the lowest 2 dimensions. e.g. if
    // ndims=4, is_vnni_format=false, format is ABCDcd. if ndims=5,
    // is_vnni_format=false, then the format is ABCDEde.
    static sc_data_format_kind_t get_2dblocking_by_dims(
            size_t ndims, bool is_weight = false, bool is_vnni_format = false);
};

namespace format_kinds {
#define SC_DEF_FMT(name, ...) \
    constexpr sc_data_format_kind_t name {__VA_ARGS__};
/* this format means continous memory format, which can be converted to
         any format*/
SC_DEF_FMT(any, 0xffffffffffffffff)
SC_DEF_FMT(A, 0)
SC_DEF_FMT(AB, 0, 1)
SC_DEF_FMT(BA, 1, 0)
SC_DEF_FMT(ABC, 0, 1, 2)
SC_DEF_FMT(ABCD, 0, 1, 2, 3)
SC_DEF_FMT(ABCDE, 0, 1, 2, 3, 4)
SC_DEF_FMT(CDBA, 2, 3, 1, 0)

// channel last format
SC_DEF_FMT(ACB, 0, 2, 1)
SC_DEF_FMT(ACDB, 0, 2, 3, 1)
SC_DEF_FMT(ACDEB, 0, 2, 3, 4, 1)

// blocked format start
SC_DEF_FMT(Aa, 0, 0)
SC_DEF_FMT(ABab, 0, 1, 0, 1)
SC_DEF_FMT(ABba, 0, 1, 1, 0)
SC_DEF_FMT(BAab, 1, 0, 0, 1)
SC_DEF_FMT(ABCb, 0, 1, 2, 1)
SC_DEF_FMT(ABCba, 0, 1, 2, 1, 0)
SC_DEF_FMT(ABCDb, 0, 1, 2, 3, 1)
SC_DEF_FMT(ABCDba, 0, 1, 2, 3, 1, 0)
SC_DEF_FMT(ABCDab, 0, 1, 2, 3, 0, 1)
SC_DEF_FMT(ABCDaba, 0, 1, 2, 3, 0, 1, 0)
SC_DEF_FMT(BACDab, 1, 0, 2, 3, 0, 1)
SC_DEF_FMT(ACDBa, 0, 2, 3, 1, 0)
SC_DEF_FMT(ACDEBa, 0, 2, 3, 4, 1, 0)
SC_DEF_FMT(BACDba, 1, 0, 2, 3, 1, 0)
SC_DEF_FMT(BACDEba, 1, 0, 2, 3, 4, 1, 0)

// for bert
SC_DEF_FMT(ABDCcd, 0, 1, 3, 2, 2, 3)
SC_DEF_FMT(ABDCcdc, 0, 1, 3, 2, 2, 3, 2)
SC_DEF_FMT(ABCDdcd, 0, 1, 2, 3, 3, 2, 3)
SC_DEF_FMT(ABCDEb, 0, 1, 2, 3, 4, 1)
SC_DEF_FMT(ABCDEba, 0, 1, 2, 3, 4, 1, 0)
SC_DEF_FMT(BACDEab, 1, 0, 2, 3, 4, 0, 1)

// vnni format
SC_DEF_FMT(KCSckc, 0, 1, 2, 1, 0, 1)
SC_DEF_FMT(KCRSckc, 0, 1, 2, 3, 1, 0, 1)
SC_DEF_FMT(KCDRSckc, 0, 1, 2, 3, 4, 1, 0, 1)
SC_DEF_FMT(CKRSkck, 1, 0, 2, 3, 0, 1, 0)
SC_DEF_FMT(CKDRSkck, 1, 0, 2, 3, 4, 0, 1, 0)
SC_DEF_FMT(NKknk, 1, 0, 0, 1, 0)
SC_DEF_FMT(BNKknk, 1, 0, 0, 1, 0)
SC_DEF_FMT(KNknk, 0, 1, 0, 1, 0)

// used for bertBMM
SC_DEF_FMT(ACBD, 0, 2, 1, 3)
SC_DEF_FMT(ABCDdc, 0, 1, 2, 3, 3, 2)
SC_DEF_FMT(ABCDcd, 0, 1, 2, 3, 2, 3)
SC_DEF_FMT(ACBDdc, 0, 2, 1, 3, 3, 2)
SC_DEF_FMT(ACBDcd, 0, 2, 1, 3, 2, 3)
SC_DEF_FMT(ACBDcdc, 0, 2, 1, 3, 2, 3, 2)

constexpr auto NCHW = ABCD, NHWC = ACDB, KCRS = ABCD, NKHW = ABCD, MK = AB,
               KN = AB, NK = BA, MN = AB, NCHWc = ABCDb, NCHWnc = ABCDab,
               NCHWcn = ABCDba, NCHWncn = ABCDaba, NKHWk = ABCDb,
               KCRSck = ABCDba, MKmk = ABab, NKkn = BAab, MNmn = ABab,
               NCDHW = ABCDE, NDHWC = ACDEB, KCDRS = ABCDE, NCDHWc = ABCDEb,
               KCDRSck = ABCDEba, CKRSkc = BACDab, CKDRSkc = BACDEab,
               NHWCn = ACDBa, NDHWCn = ACDEBa, CKRSck = BACDba,
               CKDRSck = BACDEba, NSC = ACB, NCS = ABC, NCSc = ABCb, KCS = ABC,
               KCSck = ABCba;

#undef SC_DEF_FMT
}; // namespace format_kinds
struct SC_API sc_data_format_t {
    using blocking_t = std::array<int, 4>;
    sc_data_format_t() : format_code_(format_kinds::any), blocks_ {0} {}
    constexpr sc_data_format_t(
            sc_data_format_kind_t format_code, const blocking_t &blocks = {0})
        : format_code_(format_code), blocks_(blocks) {}

    sc_data_format_t(const std::vector<int> &storage_args,
            const blocking_t &blocks = {0})
        : format_code_(storage_args), blocks_(blocks) {}

    bool operator==(const sc_data_format_t &other) const {
        return format_code_ == other.format_code_ && blocks_ == other.blocks_;
    }
    bool operator!=(const sc_data_format_t &other) const {
        return !(*this == other);
    }
    bool is_convertible(const sc_data_format_t &other) const;

    bool is_blocking() const;

    bool is_plain() const;

    bool is_channel_last() const;

    bool is_any() const;

    sc_data_format_t to_plain() const;

    sc_data_format_t to_channel_last() const;

    sc_format_category get_format_category() const;

    constexpr static inline sc_data_format_t NCHW() {
        return sc_data_format_t(format_kinds::NCHW);
    }
    constexpr static inline sc_data_format_t NHWC() {
        return sc_data_format_t(format_kinds::NHWC);
    }
    constexpr static inline sc_data_format_t NCHWc(int c) {
        return sc_data_format_t(format_kinds::NCHWc, {c});
    }
    constexpr static inline sc_data_format_t KCRS() {
        return sc_data_format_t(format_kinds::KCRS);
    }
    constexpr static inline sc_data_format_t KCRSck(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSck, {c, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KCRSck2c(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSckc, {c, k, 2});
    }
    constexpr static inline sc_data_format_t KCRSck4c(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSckc, {c, k, 4});
    }
    constexpr static inline sc_data_format_t NCS() {
        return sc_data_format_t(format_kinds::NCS);
    }
    constexpr static inline sc_data_format_t NSC() {
        return sc_data_format_t(format_kinds::NSC);
    }
    constexpr static inline sc_data_format_t NCSc(int c) {
        return sc_data_format_t(format_kinds::NCSc, {c});
    }
    constexpr static inline sc_data_format_t KCS() {
        return sc_data_format_t(format_kinds::KCS);
    }
    constexpr static inline sc_data_format_t KCSck(int c, int k) {
        return sc_data_format_t(format_kinds::KCSck, {c, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KCSck2c(int c, int k) {
        return sc_data_format_t(format_kinds::KCSckc, {c, k, 2});
    }
    constexpr static inline sc_data_format_t KCSck4c(int c, int k) {
        return sc_data_format_t(format_kinds::KCSckc, {c, k, 4});
    }
    constexpr static inline sc_data_format_t MK() {
        return sc_data_format_t(format_kinds::MK);
    }
    constexpr static inline sc_data_format_t MKmk(int m, int k) {
        return sc_data_format_t(format_kinds::MKmk, {m, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KN() {
        return sc_data_format_t(format_kinds::KN);
    }
    constexpr static inline sc_data_format_t NK() {
        return sc_data_format_t(format_kinds::NK);
    }
    constexpr static inline sc_data_format_t NKkn(int k, int n) {
        return sc_data_format_t(format_kinds::NKkn, {k, n, 0, 0});
    }
    constexpr static inline sc_data_format_t NKkn2k(int k, int n) {
        return sc_data_format_t(format_kinds::NKknk, {k, n, 2});
    }
    constexpr static inline sc_data_format_t NKkn4k(int k, int n) {
        return sc_data_format_t(format_kinds::NKknk, {k, n, 4});
    }
    constexpr static inline sc_data_format_t KNkn4k(int k, int n) {
        return sc_data_format_t(format_kinds::KNknk, {k, n, 4});
    }
    constexpr static inline sc_data_format_t KNkn2k(int k, int n) {
        return sc_data_format_t(format_kinds::KNknk, {k, n, 2});
    }
    constexpr static inline sc_data_format_t NCDHW() {
        return sc_data_format_t(format_kinds::NCDHW);
    }
    constexpr static inline sc_data_format_t NDHWC() {
        return sc_data_format_t(format_kinds::NDHWC);
    }
    constexpr static inline sc_data_format_t NCDHWc(int c) {
        return sc_data_format_t(format_kinds::NCDHWc, {c});
    }
    constexpr static inline sc_data_format_t NCHWnc(int n, int c) {
        return sc_data_format_t(format_kinds::NCHWnc, {n, c});
    }
    constexpr static inline sc_data_format_t NCHWcn(int c, int n) {
        return sc_data_format_t(format_kinds::NCHWcn, {c, n});
    }
    constexpr static inline sc_data_format_t NCHWnc2n(int n, int c) {
        return sc_data_format_t(format_kinds::NCHWncn, {n, c, 2});
    }
    constexpr static inline sc_data_format_t KCDRS() {
        return sc_data_format_t(format_kinds::KCDRS);
    }
    constexpr static inline sc_data_format_t KCDRSck(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSck, {c, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KCDRSck2c(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSckc, {c, k, 2, 0});
    }
    constexpr static inline sc_data_format_t KCDRSck4c(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSckc, {c, k, 4, 0});
    }
    constexpr static inline sc_data_format_t CKRSkc(int k, int c) {
        return sc_data_format_t(format_kinds::CKRSkc, {k, c, 0, 0});
    }
    constexpr static inline sc_data_format_t CKRSkc2k(int k, int c) {
        return sc_data_format_t(format_kinds::CKRSkck, {k, c, 2});
    }
    constexpr static inline sc_data_format_t CKDRSkc(int k, int c) {
        return sc_data_format_t(format_kinds::CKDRSkc, {k, c, 0, 0});
    }
    constexpr static inline sc_data_format_t CKDRSkc2k(int k, int c) {
        return sc_data_format_t(format_kinds::CKDRSkck, {k, c, 2});
    }
    constexpr static inline sc_data_format_t NHWCn(int n) {
        return sc_data_format_t(format_kinds::NHWCn, {n});
    }
    constexpr static inline sc_data_format_t NDHWCn(int n) {
        return sc_data_format_t(format_kinds::NDHWCn, {n});
    }
    constexpr static inline sc_data_format_t CKRSck(int c, int k) {
        return sc_data_format_t(format_kinds::CKRSck, {c, k});
    }
    constexpr static inline sc_data_format_t CKDRSck(int c, int k) {
        return sc_data_format_t(format_kinds::CKDRSck, {c, k});
    }

    sc_data_format_kind_t format_code_;
    // The blocking numbers. It stores the blocking of the blocking axis in
    // the format_code_ from left to right. At most 4 blocking numbers can be
    // stored. Unused slots should be 0. For example, for format NK16k8n4k, the
    // blocks_ should be {16,8,4,0}. std::vector is unnecessary for block info.
    // And in g++, sizeof(vector<int>)==24, while static array only takes 16
    // bytes
    blocking_t blocks_;
    int get_blocks_size() const;
    bool is_same_format_kind(const sc_data_format_t &input_format) const;

    // {plain_axis, block}
    std::unordered_map<int, std::vector<int>> get_blocked_axis() const;

    static sc_dims get_reordered_shapes(const sc_dims &input_shapes,
            const sc_data_format_t &input_format,
            const sc_data_format_t &output_format);
    // given plain shapes and the data format, gets the real blocking shapes
    static sc_dims get_blocking_shapes(
            const sc_dims &plain_shapes, const sc_data_format_t &format);
    // given real blocking shapes and the data format, infers plain shapes. Note
    // that if there is padding when converting plain shapes and format to
    // blocking shapes, we cannot infer the original plain shapes from the
    // padded blocking shapes and the format
    static sc_dims get_padded_plain_shapes(
            const sc_dims &real_shapes, const sc_data_format_t &format);

    // gets an N-D plain format
    static sc_data_format_t get_plain_by_dims(size_t shape_size) {
        return sc_data_format_t(
                sc_data_format_kind_t::get_plain_by_dims(shape_size));
    }

    runtime::dispatch_key to_runtime() const;

    void to_string(std::ostream &os) const;
};
struct sc_data_format_cmper_t {
    bool operator()(
            const sc_data_format_t &key0, const sc_data_format_t &key1) const;
};

SC_INTERNAL_API std::ostream &operator<<(
        std::ostream &os, const sc_data_format_t &in);

// if has block on dynamic plain shapes
bool is_dynamic_blocking(const sc_dims &shapes, const sc_data_format_t &format);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::gc::sc_data_format_t> {
    std::size_t operator()(
            const dnnl::impl::graph::gc::sc_data_format_t &k) const;
};
} // namespace std

#endif
