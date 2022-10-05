/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/conv/config_lookup_table.hpp"

#include <string>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

ngen::HW to_hw(const std::string &s) {
    using namespace ir_utils;
#define CASE(name) \
    if (s == #name || s == to_lower(#name)) return ngen::HW::name;
    CASE(XeHP)
    CASE(XeHPG)
    CASE(XeHPC)
#undef CASE
    ir_error_not_expected();
    return ngen::HW::Unknown;
}

int_filter_t::int_filter_t(const std::string &s) {
    cmp_op_ = op_kind_t::_eq;
    if (s.empty()) {
        value_ = 0;
        return;
    }
    auto end = s.size();
    auto last = s[end - 1];
    if (last == '+') {
        cmp_op_ = op_kind_t::_ge;
        end--;
    }
    value_ = std::stoi(s.substr(0, end));
}

bool int_filter_t::matches(int value) const {
    switch (cmp_op_) {
        case op_kind_t::_eq: return value == value_;
        case op_kind_t::_le: return value <= value_;
        case op_kind_t::_ge: return value >= value_;
        case op_kind_t::_lt: return value < value_;
        case op_kind_t::_gt: return value > value_;
        default: ir_error_not_expected();
    }
    return false;
}

type_filter_t::type_filter_t(const std::string &s) {
    for (size_t pos = 0;;) {
        bool found = false;
        for (auto &p : all_patterns()) {
            if (try_parse(s, pos, p)) {
                found = true;
                break;
            }
        }
        if (!found) {
            ir_assert(pos == s.size()) << s;
            break;
        }
    }
}

bool type_filter_t::matches(const std::vector<data_type_t> &values) const {
    ir_assert(values.size() == patterns_.size());
    for (size_t i = 0; i < values.size(); i++) {
        auto &ptrn = patterns_[i];
        if (ptrn == "*") continue;
        if (ptrn == "x8") {
            if (!utils::one_of(values[i], data_type::s8, data_type::u8))
                return false;
        } else {
            ir_error_not_expected() << ptrn;
        }
    }
    return true;
}

bool type_filter_t::try_parse(
        const std::string &s, size_t &pos, const std::string &pattern) {
    if (pos + pattern.size() > s.size()) return false;
    if (!std::equal(pattern.begin(), pattern.end(), s.begin() + pos))
        return false;
    patterns_.push_back(s.substr(pos, pattern.size()));
    pos = pos + pattern.size();
    return true;
}

std::vector<std::string> &type_filter_t::all_patterns() {
    static std::vector<std::string> ret = {
            "x8",
            "*",
    };
    return ret;
}
conv_problem_filter_t::conv_problem_filter_t(const std::string &s) {
    auto parts = ir_utils::split(s, " ");
    for (auto &part : parts) {
        auto sub_parts = ir_utils::split(part, "=");
        ir_assert(sub_parts.size() == 2) << part;
        auto &name = sub_parts[0];
        auto &value = sub_parts[1];
        if (name == "hw") {
            hw_ = to_hw(value);
        } else if (name == "cfg") {
            type_filter_ = type_filter_t(value);
        } else if (name == "dir") {
            dir_ = value;
        } else if (name == "desc") {
            desc_ = value;
        } else if (name == "mb") {
            mb_filter_ = int_filter_t(value);
        } else if (name == "post_ops") {
            post_ops_ = value;
        } else {
            ir_error_not_expected() << part;
        }
    }
}

bool conv_problem_filter_t::matches(
        const conv_problem_t &prb, const hw_config_t &hw_cfg) const {
    if (hw_cfg.hw() != hw_) return false;
    if (!matches_dir(prb)) return false;
    if (!type_filter_.matches(
                {prb.src_data_type, prb.wei_data_type, prb.dst_data_type}))
        return false;
    if (!mb_filter_.matches(prb.mb)) return false;
    if (!matches_desc(prb)) return false;
    if (!matches_post_ops(prb)) return false;
    return true;
}

bool conv_problem_filter_t::matches_dir(const conv_problem_t &prb) const {
    if (dir_.empty()) return true;
    if (dir_ == "fwd") {
        return prb.is_fwd;
    } else if (dir_ == "bwd_d") {
        return prb.is_bwd_d;
    } else if (dir_ == "bwd_w") {
        return prb.is_bwd_w;
    } else {
        ir_error_not_expected() << dir_;
    }
    return false;
}

bool conv_problem_filter_t::matches_desc(const conv_problem_t &prb) const {
    return prb.desc_str(/*print_mb=*/false) == desc_;
}

bool conv_problem_filter_t::matches_post_ops(const conv_problem_t &prb) const {
    if (post_ops_ == "sum") return prb.with_sum;
    ir_assert(post_ops_.empty()) << post_ops_;
    return !prb.with_sum;
}

conv_config_lookup_table_t::conv_config_lookup_table_t() {
    // clang-format off
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic2048iw49oc512ow49kw1pw0", "T=oc8mb1osp4 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic128iw784oc512ow784kw1pw0 post_ops=sum", "T=oc8mb1osp4 s=x2.g1.v1 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256ih28oc256oh14kh3sh2ph1", "T=oc8mb1osp4 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic64iw3136oc256ow3136kw1pw0", "T=oc2mb1osp4 s=x3.g2.v2 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512ih28oc1024oh14kh1sh2ph0", "T=oc8mb1osp4 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic1024ih14oc2048oh7kh1sh2ph0", "T=oc8mb4osp1 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256iw196oc1024ow196kw1pw0 post_ops=sum", "T=oc8mb1osp4 s=x3.g2.v2 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256iw3136oc64ow3136kw1pw0", "T=oc4mb1osp8 s=x1.g1.v0 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256ih14oc256oh14kh3ph1", "T=oc8mb4osp1 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512iw49oc2048ow49kw1pw0 post_ops=sum", "T=oc8mb4osp1 s=x3.g1.v2 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic64ih56oc64oh56kh3ph1", "T=oc2mb1osp8 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic128ih28oc128oh28kh3ph1", "T=oc4mb1osp8 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic1024iw196oc256ow196kw1pw0", "T=oc8mb1osp4 s=x3.g1.v3 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512ih14oc512oh7kh3sh2ph1", "T=oc8mb4osp1 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic64iw3136oc64ow3136kw1pw0", "T=oc4mb1osp8 s=x1.g1.v0 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512ih7oc512oh7kh3ph1", "T=oc4mb4osp1 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic128ih56oc128oh28kh3sh2ph1", "T=oc4mb1osp8 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256ih56oc512oh28kh1sh2ph0", "T=oc8mb1osp4 s=x3.g2.v2 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic3ih224oc64oh112kh7sh2ph3", "T=oc2mb1osp8 s=x3.g1.v2 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512iw784oc128ow784kw1pw0", "T=oc4mb1osp2 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic64iw3136oc256ow3136kw1pw0 post_ops=sum", "T=oc4mb1osp2 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic1024iw196oc512ow196kw1pw0", "T=oc8mb1osp4 s=x3.g2.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic256iw3136oc128ow3136kw1pw0", "T=oc4mb1osp2 s=x3.g1.v4 c=0");
        add("hw=xehpg dir=fwd cfg=x8x8* mb=128+ desc=ic512iw784oc256ow784kw1pw0", "T=oc8mb1osp4 s=x3.g2.v3 c=0");
    // clang-format on
}

const char *conv_config_lookup_table_t::find(const conv_config_t &cfg) const {
    auto key = cfg.prb().desc_str(/*print_mb=*/false);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    for (auto &e : it->second) {
        if (e.filter.matches(cfg.prb(), cfg.hw_cfg())) return e.s_params;
    }
    return nullptr;
}

void conv_config_lookup_table_t::add(const char *s_prb, const char *s_params) {
    conv_problem_filter_t filter(s_prb);
    map_[filter.key()].push_back(entry_t {filter, s_params});
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
