/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/key.hpp"

#include <functional>
#include <limits>
#include <sstream>
#include <string>

#include "common/utils.hpp"
#include "gpu/intel/jit/conv/config.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class key_type_kind_t {
    undef,
    any,
    s8,
    u8,
    x8, // s8 or u8
    bf16,
    f16,
    x16, // f16 or bf16
    f32,
    s32,
    tf32,
    f64,
    bf8,
    f8_e5m2 = bf8,
    hf8,
    f8_e4m3 = hf8,
    xf8, // bf8 or hf8
    _max,
};

static auto key_type_kind_names = nstl::to_array({
        make_enum_name(key_type_kind_t::undef, "undef"),
        make_enum_name(key_type_kind_t::any, "any"),
        make_enum_name(key_type_kind_t::s8, "s8"),
        make_enum_name(key_type_kind_t::u8, "u8"),
        make_enum_name(key_type_kind_t::x8, "x8"),
        make_enum_name(key_type_kind_t::bf16, "bf16"),
        make_enum_name(key_type_kind_t::f16, "f16"),
        make_enum_name(key_type_kind_t::x16, "x16"),
        make_enum_name(key_type_kind_t::bf8, "bf8"),
        make_enum_name(key_type_kind_t::hf8, "hf8"),
        make_enum_name(key_type_kind_t::xf8, "xf8"),
        make_enum_name(key_type_kind_t::f32, "f32"),
        make_enum_name(key_type_kind_t::s32, "s32"),
        make_enum_name(key_type_kind_t::tf32, "tf32"),
        make_enum_name(key_type_kind_t::f64, "f64"),
});

GPU_DEFINE_PARSE_ENUM(key_type_kind_t, key_type_kind_names)

namespace {

template <typename KindT>
struct key_kind_traits_t {
    static bool matches(KindT a, KindT b) { return a == b; }
};

fma_kind_t to_key(fma_kind_t fma) {
    switch (fma) {
        case fma_kind_t::mad:
        case fma_kind_t::dp4a:
        case fma_kind_t::dpas: return fma;
        case fma_kind_t::dpasw: return fma_kind_t::dpas;
        default: gpu_error_not_expected(); return fma_kind_t::undef;
    }
}

key_type_kind_t to_type_kind(data_type_t dt) {
#define CASE(name) \
    case data_type::name: return key_type_kind_t::name
    switch ((int)dt) {
        CASE(s8);
        CASE(u8);
        CASE(bf16);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(tf32);
        CASE(f64);
        default: gpu_error_not_expected(); return key_type_kind_t::undef;
    }
#undef CASE
}

key_type_kind_t to_filter(key_type_kind_t kind) {
    switch (kind) {
        case key_type_kind_t::any:
        case key_type_kind_t::f32:
        case key_type_kind_t::s32:
        case key_type_kind_t::tf32:
        case key_type_kind_t::f64:
        case key_type_kind_t::undef: return kind;
        case key_type_kind_t::s8:
        case key_type_kind_t::u8:
        case key_type_kind_t::x8: return key_type_kind_t::x8;
        case key_type_kind_t::f16:
        case key_type_kind_t::bf16:
        case key_type_kind_t::x16: return key_type_kind_t::x16;
        case key_type_kind_t::bf8:
        case key_type_kind_t::hf8:
        case key_type_kind_t::xf8: return key_type_kind_t::xf8;
        default: gpu_error_not_expected();
    }
    return key_type_kind_t::undef;
}

template <>
struct key_kind_traits_t<key_type_kind_t> {
    static bool matches(key_type_kind_t filter, key_type_kind_t other) {
        gpu_assert(filter == to_filter(filter));
        if (filter == key_type_kind_t::any) return true;
        return filter == to_filter(other);
    }
};

template <typename KindT>
struct subkey_t {
    KindT kind;

    using traits_t = key_kind_traits_t<KindT>;

    subkey_t() = default;
    subkey_t(KindT kind) : kind(kind) {}

    bool operator==(const subkey_t &other) const { return kind == other.kind; }

    bool matches(const subkey_t &other) const {
        return traits_t::matches(kind, other.kind);
    }

    size_t get_hash() const { return ir_utils::get_hash(kind); }

    void stringify(std::ostream &out) const { out << to_string(kind); }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        kind = to_enum<KindT>(s);
    }

    std::string str() const { return to_string(kind); }

    IR_DEFINE_DUMP()
};

using key_fma_t = subkey_t<fma_kind_t>;
using key_prop_t = subkey_t<prop_kind_t>;
using key_type_t = subkey_t<key_type_kind_t>;

struct key_hw_t {
    ngen::HW hw = ngen::HW::Unknown;
    ngen::ProductFamily family = ngen::ProductFamily::Unknown;

    key_hw_t() = default;
    key_hw_t(ngen::HW hw, ngen::ProductFamily family)
        : hw(hw), family(family) {}

    bool with_family() const { return family != ngen::ProductFamily::Unknown; }

    bool matches(const key_hw_t &other) const {
        if (hw != other.hw) return false;
        if (!with_family()) return true;
        return family == other.family;
    }

    bool operator==(const key_hw_t &other) const {
        return (hw == other.hw) && (family == other.family);
    }

    size_t get_hash() const { return ir_utils::get_hash(hw, family); }

    void stringify(std::ostream &out) const {
        out << ir_utils::to_lower(jit::to_string(hw));
        if (with_family())
            out << ":" << ir_utils::to_lower(jit::to_string(family));
    }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        auto parts = gpu_utils::split(s, ":");
        gpu_assert(parts.size() <= 2);
        hw = to_enum<ngen::HW>(parts[0]);
        family = (parts.size() > 1 ? to_enum<ngen::ProductFamily>(parts[1])
                                   : ngen::ProductFamily::Unknown);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << jit::to_string(hw);
        if (with_family()) oss << ":" << jit::to_string(family);
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct key_type_info_t {
    key_type_t src;
    key_type_t wei;
    key_type_t dst;

    key_type_info_t() = default;
    key_type_info_t(const conv_config_t &cfg) {
        auto &prb = cfg.prb();
        auto src_type = prb.a_data_type;
        auto wei_type = prb.b_data_type;
        auto dst_type = prb.c_data_type;
        if (prb.is_bwd_d) std::swap(src_type, dst_type);
        if (prb.is_bwd_w) std::swap(wei_type, dst_type);
        src = to_type_kind(src_type);
        wei = to_type_kind(wei_type);
        dst = to_type_kind(dst_type);
    }

    key_type_info_t to_filter(prop_kind_t prop) const {
        auto ret = *this;
        ret.src = key_type_t(jit::to_filter(src.kind));
        ret.wei = key_type_t(jit::to_filter(wei.kind));
        ret.dst = key_type_t(jit::to_filter(dst.kind));
        auto any_type = key_type_t(key_type_kind_t::any);
        switch (prop) {
            case prop_kind::forward: ret.dst = any_type; break;
            case prop_kind::backward_data: ret.src = any_type; break;
            case prop_kind::backward_weights: ret.wei = any_type; break;
            default: gpu_error_not_expected();
        }
        return ret;
    }

    bool operator==(const key_type_info_t &other) const {
        return (src == other.src) && (wei == other.wei) && (dst == other.dst);
    }

    bool matches(const key_type_info_t &other) const {
        return src.matches(other.src) && wei.matches(other.wei)
                && dst.matches(other.dst);
    }

    size_t get_hash() const { return ir_utils::get_hash(src, wei, dst); }

    void stringify(std::ostream &out) const {
        src.stringify(out);
        out << ":";
        wei.stringify(out);
        out << ":";
        dst.stringify(out);
    }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        auto parts = gpu_utils::split(s, ":");
        gpu_assert(parts.size() == 3);
        src = key_type_t(to_enum<key_type_kind_t>(parts[0]));
        wei = key_type_t(to_enum<key_type_kind_t>(parts[1]));
        dst = key_type_t(to_enum<key_type_kind_t>(parts[2]));
    }

    std::string str() const {
        std::ostringstream oss;
        if (src == wei && src == dst) {
            oss << src.str();
        } else {
            oss << src.str() << wei.str() << dst.str();
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

bool is_mb_blocked(const layout_t &layout) {
    dim_t blk
            = layout.inner_block(0, /*skip_outer=*/true, /*inner_only=*/false);
    return blk > 1;
}

struct key_mb_t {
    bool is_blocked = false;
    dim_t value = 0;

    key_mb_t() = default;
    key_mb_t(const conv_config_t &cfg, prop_kind_t prop) {
        auto &prb = cfg.prb();
        auto src_blocked = is_mb_blocked(cfg.src_layout().compute());
        auto dst_blocked = is_mb_blocked(cfg.dst_layout().compute());
        value = prb.mb;
        switch (prop) {
            case prop_kind::forward: is_blocked = src_blocked; break;
            case prop_kind::backward_data: is_blocked = dst_blocked; break;
            case prop_kind::backward_weights:
                is_blocked = src_blocked && dst_blocked;
                break;
            default: gpu_error_not_expected();
        }
    }

    bool operator==(const key_mb_t &other) const {
        return (is_blocked == other.is_blocked) && (value == other.value);
    }

    bool matches(const key_mb_t &other) const {
        if (is_blocked != other.is_blocked) return false;
        return value <= other.value;
    }

    size_t get_hash() const { return ir_utils::get_hash(is_blocked); }

    void stringify(std::ostream &out) const {
        out << "mb" << value;
        if (is_blocked) out << "b";
    }

    void parse(std::istream &in) {
        stream_match(in, "mb");
        value = stream_parse<int>(in);
        stream_try_match(in, "+");
        is_blocked = stream_try_match(in, "b");
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "mb" << value;
        if (is_blocked) oss << "(blocked)";
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct key_desc_t {
    std::string desc;

    key_desc_t() = default;
    key_desc_t(const std::string &desc) : desc(desc) {}

    bool operator==(const key_desc_t &other) const {
        return desc == other.desc;
    }

    bool matches(const key_desc_t &other) const { return operator==(other); }

    size_t get_hash() const { return std::hash<std::string>()(desc); }

    void stringify(std::ostream &out) const { out << desc; }

    void parse(std::istream &in) { desc = stream_parse<std::string>(in); }

    std::string str() const { return desc; }

    IR_DEFINE_DUMP()
};

} // namespace

class conv_key_impl_t {
public:
    conv_key_impl_t() = default;

    conv_key_impl_t(const key_hw_t &hw, const key_fma_t &fma,
            const key_prop_t &prop, const key_type_info_t &type_info,
            const key_mb_t &mb, const key_desc_t &desc)
        : hw_(hw)
        , fma_(fma)
        , prop_(prop)
        , type_info_(type_info)
        , mb_(mb)
        , desc_(desc) {}

    const std::string &desc_str() const { return desc_.desc; }

    conv_key_impl_t to_filter() const {
        conv_key_impl_t ret = *this;
        ret.type_info_ = type_info_.to_filter(prop_.kind);
        return ret;
    }

    dim_t distance(const conv_key_impl_t &other) const {
        int max_dist = std::numeric_limits<int>::max();
        if (!matches(other)) return max_dist;
        // Here this object is a filter, other object is a non-filter.
        // matches(other) ensures that mb_.value <= other.mb_.value.
        // Example:
        //   Key     : mb512
        //   Filter A: mb128+ (distance: 384)
        //   Filter B: mb256+ (distance: 256) <- smaller distance, preferred.
        dim_t dist = other.mb_.value - mb_.value;
        auto f1 = hw_.family;
        auto f2 = other.hw_.family;
        if (f1 != f2) {
            const int large_dist = (1 << 20);
            dist += large_dist;
            if (!utils::one_of(ngen::ProductFamily::Unknown, f1, f2))
                dist += large_dist;
        }
        return dist;
    }

    bool operator==(const conv_key_impl_t &other) const {
        return (hw_ == other.hw_) && (fma_ == other.fma_)
                && (prop_ == other.prop_) && (type_info_ == other.type_info_)
                && (mb_ == other.mb_) && (desc_ == other.desc_);
    }

    bool matches(const conv_key_impl_t &other) const {
        return hw_.matches(other.hw_) && fma_.matches(other.fma_)
                && prop_.matches(other.prop_)
                && type_info_.matches(other.type_info_)
                && mb_.matches(other.mb_) && desc_.matches(other.desc_);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(hw_, fma_, prop_, type_info_, mb_, desc_);
    }

    void stringify(std::ostream &out) const {
        hw_.stringify(out);
        out << " ";
        fma_.stringify(out);
        out << " ";
        prop_.stringify(out);
        out << " ";
        type_info_.stringify(out);
        out << " ";
        mb_.stringify(out);
        out << " ";
        desc_.stringify(out);
    }

    void parse(std::istream &in) {
        hw_.parse(in);
        fma_.parse(in);
        prop_.parse(in);
        type_info_.parse(in);
        mb_.parse(in);
        desc_.parse(in);
    }

    std::string str(bool csv = false) const {
        std::ostringstream oss;
        oss << hw_;
        oss << "," << fma_;
        oss << "," << prop_;
        oss << "," << type_info_;
        if (csv) {
            oss << ","
                << "mb" << mb_.value << desc_;
        } else {
            oss << "," << mb_;
            oss << "," << desc_;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    key_hw_t hw_;
    key_fma_t fma_;
    key_prop_t prop_;
    key_type_info_t type_info_;
    key_mb_t mb_;
    key_desc_t desc_;
};

conv_key_t::conv_key_t(const conv_config_t &cfg, bool make_filter) {
    auto &prb = cfg.prb();
    auto hw = key_hw_t(cfg.hw().to_ngen(), cfg.hw().product_family());
    auto fma = key_fma_t(to_key(cfg.fma_kind()));
    auto prop = key_prop_t(prb.prop_kind());
    auto type_info = key_type_info_t(cfg);
    auto mb = key_mb_t(cfg, prop.kind);
    auto desc = key_desc_t(prb.desc_str(/*print_mb=*/false));
    auto impl = conv_key_impl_t(hw, fma, prop, type_info, mb, desc);
    if (make_filter) impl = impl.to_filter();
    impl_ = std::make_shared<conv_key_impl_t>(impl);
}

conv_key_t conv_key_t::to_filter() const {
    if (!impl_) return conv_key_t();
    conv_key_t ret;
    ret.impl_ = std::make_shared<conv_key_impl_t>(impl_->to_filter());
    return ret;
}

const std::string &conv_key_t::desc() const {
    gpu_assert(impl_);
    return impl_->desc_str();
}

dim_t conv_key_t::distance(const conv_key_t &other) const {
    gpu_assert(impl_ && other.impl_);
    return impl_->distance(*other.impl_);
}

bool conv_key_t::operator==(const conv_key_t &other) const {
    if (!impl_ || !other.impl_) return impl_ == other.impl_;
    return impl_->operator==(*other.impl_);
}

bool conv_key_t::matches(const conv_key_t &other) const {
    if (!impl_ || !other.impl_) return impl_ == other.impl_;
    return impl_->matches(*other.impl_);
}

size_t conv_key_t::get_hash() const {
    return impl_ ? impl_->get_hash() : 0;
}

void conv_key_t::stringify(std::ostream &out) const {
    gpu_assert(impl_);
    impl_->stringify(out);
}

void conv_key_t::parse(std::istream &in) {
    impl_ = std::make_shared<conv_key_impl_t>();
    impl_->parse(in);
}

std::string conv_key_t::str(bool csv) const {
    if (!impl_) return "(nil)";
    return impl_->str(csv);
}

std::vector<std::string> conv_key_t::csv_keys() {
    return {"hw", "fma", "prop", "cfg", "desc"};
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
