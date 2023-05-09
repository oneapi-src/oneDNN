/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/conv/key.hpp"

#include <functional>
#include <limits>
#include <sstream>
#include <string>

#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

namespace {

template <typename KindT>
struct key_kind_traits_t {
    static bool supports_filter() { return false; }
    static bool is_filter(KindT kind) { return false; }
    static bool matches(KindT a, KindT b) { return a == b; }
};

enum class key_hw_kind_t {
    undef,
    gen9,
    gen12lp,
    xehp,
    xehpg,
    xehpc,
};

std::string to_string(key_hw_kind_t kind) {
#define CASE(name) \
    case key_hw_kind_t::name: return #name
    switch (kind) {
        CASE(undef);
        CASE(gen9);
        CASE(gen12lp);
        CASE(xehp);
        CASE(xehpg);
        CASE(xehpc);
        default: ir_error_not_expected();
    }
#undef CASE
    return {};
}

key_hw_kind_t to_hw_kind(ngen::HW hw) {
    switch (hw) {
        case ngen::HW::Gen9: return key_hw_kind_t::gen9;
        case ngen::HW::Gen12LP: return key_hw_kind_t::gen12lp;
        case ngen::HW::XeHP: return key_hw_kind_t::xehp;
        case ngen::HW::XeHPG: return key_hw_kind_t::xehpg;
        case ngen::HW::XeHPC: return key_hw_kind_t::xehpc;
        default: ir_error_not_expected(); return key_hw_kind_t::undef;
    }
}

enum class key_fma_kind_t {
    undef,
    mad,
    dp4a,
    dpas,
};

std::string to_string(key_fma_kind_t kind) {
#define CASE(name) \
    case key_fma_kind_t::name: return #name
    switch (kind) {
        CASE(undef);
        CASE(mad);
        CASE(dpas);
        default: ir_error_not_expected();
    }
#undef CASE
    return {};
}

key_fma_kind_t to_fma_kind(fma_kind_t fma) {
    switch (fma) {
        case fma_kind_t::mad: return key_fma_kind_t::mad;
        case fma_kind_t::dp4a: return key_fma_kind_t::dp4a;
        case fma_kind_t::dpas:
        case fma_kind_t::dpasw: return key_fma_kind_t::dpas;
        default: ir_error_not_expected(); return key_fma_kind_t::undef;
    }
}

enum class key_prop_kind_t {
    undef,
    fwd,
    bwd_d,
    bwd_w,
};

std::string to_string(key_prop_kind_t kind) {
#define CASE(name) \
    case key_prop_kind_t::name: return #name
    switch (kind) {
        CASE(undef);
        CASE(fwd);
        CASE(bwd_d);
        CASE(bwd_w);
        default: ir_error_not_expected();
    }
#undef CASE
    return {};
}

key_prop_kind_t to_prop_kind(prop_kind_t kind) {
    switch (kind) {
        case prop_kind::forward_inference:
        case prop_kind::forward_training: return key_prop_kind_t::fwd;
        case prop_kind::backward_data: return key_prop_kind_t::bwd_d;
        case prop_kind::backward_weights: return key_prop_kind_t::bwd_w;
        default: ir_error_not_expected(); return key_prop_kind_t::undef;
    }
}

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
};

std::string to_string(key_type_kind_t kind) {
#define CASE(name) \
    case key_type_kind_t::name: return #name
    switch (kind) {
        CASE(undef);
        CASE(any);
        CASE(s8);
        CASE(u8);
        CASE(x8);
        CASE(bf16);
        CASE(f16);
        CASE(x16);
        CASE(f32);
        CASE(s32);
        CASE(tf32);
        CASE(f64);
        default: ir_error_not_expected();
    }
#undef CASE
    return {};
}

key_type_kind_t to_type_kind(data_type_t dt) {
#define CASE(name) \
    case data_type::name: return key_type_kind_t::name
    switch ((int)dt) {
        CASE(s8);
        CASE(u8);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(tf32);
        CASE(f64);
        default: ir_error_not_expected(); return key_type_kind_t::undef;
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
        default: ir_error_not_expected();
    }
    return key_type_kind_t::undef;
}

template <>
struct key_kind_traits_t<key_type_kind_t> {
    static bool supports_filter() { return true; }

    static bool is_filter(key_type_kind_t kind) {
        switch (kind) {
            case key_type_kind_t::any:
            case key_type_kind_t::x8:
            case key_type_kind_t::x16: return true;
            default: return false;
        }
    }

    static bool matches(key_type_kind_t a, key_type_kind_t b) {
        if (is_filter(a) && is_filter(b)) return a == b;
        if (!is_filter(a) && is_filter(b)) return matches(b, a);
        switch (a) {
            case key_type_kind_t::any: return true;
            case key_type_kind_t::x8:
                return utils::one_of(
                        b, key_type_kind_t::s8, key_type_kind_t::u8);
            case key_type_kind_t::x16:
                return utils::one_of(
                        b, key_type_kind_t::bf16, key_type_kind_t::f16);
            default: ir_assert(!is_filter(a)); return a == b;
        }
    }
};

template <typename KindT>
struct subkey_t {
    KindT kind;

    using traits_t = key_kind_traits_t<KindT>;

    subkey_t() = default;
    subkey_t(KindT kind) : kind(kind) {}

    bool is_filter() const { return traits_t::is_filter(kind); }

    bool operator==(const subkey_t &other) const { return kind == other.kind; }

    bool matches(const subkey_t &other) const {
        return traits_t::matches(kind, other.kind);
    }

    size_t get_hash() const {
        if (traits_t::supports_filter()) return 0;
        return ir_utils::get_hash(kind);
    }

    void serialize(std::ostream &out) const { ir_utils::serialize(kind, out); }

    void deserialize(std::istream &in) {
        kind = ir_utils::deserialize<KindT>(in);
    }

    std::string str() const { return to_string(kind); }

    IR_DEFINE_DUMP()
};

using key_hw_t = subkey_t<key_hw_kind_t>;
using key_fma_t = subkey_t<key_fma_kind_t>;
using key_prop_t = subkey_t<key_prop_kind_t>;
using key_type_t = subkey_t<key_type_kind_t>;

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

    bool is_filter() const {
        return src.is_filter() || wei.is_filter() || dst.is_filter();
    }

    key_type_info_t to_filter(key_prop_kind_t prop) const {
        auto ret = *this;
        ret.src = key_type_t(jit::to_filter(src.kind));
        ret.wei = key_type_t(jit::to_filter(wei.kind));
        ret.dst = key_type_t(jit::to_filter(dst.kind));
        auto any_type = key_type_t(key_type_kind_t::any);
        switch (prop) {
            case key_prop_kind_t::fwd: ret.dst = any_type; break;
            case key_prop_kind_t::bwd_d: ret.src = any_type; break;
            case key_prop_kind_t::bwd_w: ret.wei = any_type; break;
            default: ir_error_not_expected();
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

    void serialize(std::ostream &out) const {
        src.serialize(out);
        wei.serialize(out);
        dst.serialize(out);
    }

    void deserialize(std::istream &in) {
        src.deserialize(in);
        wei.deserialize(in);
        dst.deserialize(in);
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
    bool is_filter = false;
    bool is_blocked = false;
    int value = 0;

    key_mb_t() = default;
    key_mb_t(const conv_config_t &cfg, key_prop_kind_t prop) {
        auto &prb = cfg.prb();
        auto src_blocked = is_mb_blocked(cfg.src_layout().compute());
        auto dst_blocked = is_mb_blocked(cfg.dst_layout().compute());
        value = prb.mb;
        switch (prop) {
            case key_prop_kind_t::fwd: is_blocked = src_blocked; break;
            case key_prop_kind_t::bwd_d: is_blocked = dst_blocked; break;
            case key_prop_kind_t::bwd_w:
                is_blocked = src_blocked && dst_blocked;
                break;
            default: ir_error_not_expected();
        }
    }

    key_mb_t to_filter() const {
        auto ret = *this;
        ret.is_filter = true;
        return ret;
    }

    bool operator==(const key_mb_t &other) const {
        return (is_filter == other.is_filter)
                && (is_blocked == other.is_blocked) && (value == other.value);
    }

    bool matches(const key_mb_t &other) const {
        if (is_filter && other.is_filter) return operator==(other);
        if (!is_filter && other.is_filter) return other.matches(*this);
        if (is_blocked != other.is_blocked) return false;
        if (is_filter) return value <= other.value;
        return value == other.value;
    }

    size_t get_hash() const { return ir_utils::get_hash(is_blocked); }

    void serialize(std::ostream &out) const {
        ir_utils::serialize(is_filter, out);
        ir_utils::serialize(is_blocked, out);
        ir_utils::serialize(value, out);
    }

    void deserialize(std::istream &in) {
        is_filter = ir_utils::deserialize<bool>(in);
        is_blocked = ir_utils::deserialize<bool>(in);
        value = ir_utils::deserialize<int>(in);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "mb" << value;
        if (is_filter) oss << "+";
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

    void serialize(std::ostream &out) const { ir_utils::serialize(desc, out); }

    void deserialize(std::istream &in) {
        desc = ir_utils::deserialize<std::string>(in);
    }

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

    bool is_filter() const { return type_info_.is_filter() || mb_.is_filter; }

    conv_key_impl_t to_filter() const {
        conv_key_impl_t ret = *this;
        ret.type_info_ = type_info_.to_filter(prop_.kind);
        ret.mb_ = mb_.to_filter();
        return ret;
    }

    int distance(const conv_key_impl_t &other) const {
        ir_assert(!other.is_filter());
        int max_dist = std::numeric_limits<int>::max();
        if (!matches(other)) return max_dist;
        if (!is_filter()) return 0;
        // Here this object is a filter, other object is a non-filter.
        // matches(other) ensures that mb_.value <= other.mb_.value.
        // Example:
        //   Key     : mb512
        //   Filter A: mb128+ (distance: 384)
        //   Filter B: mb256+ (distance: 256) <- smaller distance, preferred.
        return other.mb_.value - mb_.value;
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

    bool is_desc_equal(const conv_key_impl_t &other) const {
        return desc_ == other.desc_;
    }

    size_t get_hash() const {
        return ir_utils::get_hash(hw_, fma_, prop_, type_info_, mb_, desc_);
    }

    void serialize(std::ostream &out) const {
        hw_.serialize(out);
        fma_.serialize(out);
        prop_.serialize(out);
        type_info_.serialize(out);
        mb_.serialize(out);
        desc_.serialize(out);
    }

    void deserialize(std::istream &in) {
        hw_.deserialize(in);
        fma_.deserialize(in);
        prop_.deserialize(in);
        type_info_.deserialize(in);
        mb_.deserialize(in);
        desc_.deserialize(in);
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
    auto hw = key_hw_t(to_hw_kind(cfg.hw()));
    auto fma = key_fma_t(to_fma_kind(cfg.fma_kind()));
    auto prop = key_prop_t(to_prop_kind(prb.prop_kind()));
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

int conv_key_t::distance(const conv_key_t &other) const {
    ir_assert(impl_ && other.impl_);
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

bool conv_key_t::is_desc_equal(const conv_key_t &other) const {
    if (!impl_ || !other.impl_) return impl_ == other.impl_;
    return impl_->is_desc_equal(*other.impl_);
}

size_t conv_key_t::get_hash() const {
    return impl_ ? impl_->get_hash() : 0;
}

void conv_key_t::serialize(std::ostream &out) const {
    ir_assert(impl_);
    impl_->serialize(out);
}

void conv_key_t::deserialize(std::istream &in) {
    impl_ = std::make_shared<conv_key_impl_t>();
    impl_->deserialize(in);
}

std::string conv_key_t::str(bool csv) const {
    if (!impl_) return "(nil)";
    return impl_->str(csv);
}

std::vector<std::string> conv_key_t::csv_keys() {
    return {"hw", "fma", "prop", "cfg", "desc"};
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
