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

#include <cctype>
#include <string>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::vector<std::string> split(
        const std::string &s, const std::string &delimiter) {
    size_t beg = 0;
    size_t end = 0;
    std::vector<std::string> ret;
    while (end != std::string::npos) {
        beg = (end == 0) ? 0 : end + delimiter.size();
        end = s.find(delimiter, beg);
        size_t len
                = (end == std::string::npos) ? std::string::npos : (end - beg);
        ret.push_back(s.substr(beg, len));
    }
    return ret;
}

std::string str_tolower(const std::string &s) {
    auto ret = s;
    std::transform(ret.begin(), ret.end(), ret.begin(),
            [](char c) { return std::tolower(c); });
    return ret;
}

bool str_to_bool(const std::string &s) {
    if (s == "0" || s == "false") return false;
    return true;
}

ngen::HW str_to_hw(const std::string &s) {
#define CASE(name) \
    if (s == #name || s == str_tolower(#name)) return ngen::HW::name;
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
    auto parts = split(s, " ");
    for (auto &part : parts) {
        auto sub_parts = split(part, "=");
        ir_assert(sub_parts.size() == 2) << part;
        auto &name = sub_parts[0];
        auto &value = sub_parts[1];
        if (name == "hw") {
            hw_ = str_to_hw(value);
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

slm_config_t::slm_config_t(const std::string &s) {
    auto parts = split(s, ".");
    for (auto &p : parts) {
        ir_assert(p.size() >= 2) << p;
        char name = p[0];
        int value = std::stoi(p.substr(1));
        switch (name) {
            case 'x': bufs = value; break;
            case 'g': gmem_bufs = value; break;
            case 'v': sync_version = value; break;
            default: ir_error_not_expected() << p;
        }
    }
}

tile_config_t::tile_config_t(const std::string &s) {
    int name_beg = -1;
    int value_beg = -1;
    for (int pos = 0; pos < (int)s.size() + 1; pos++) {
        bool prev_digit = pos > 0 && std::isdigit(s[pos - 1]);
        bool cur_digit = pos < (int)s.size() && std::isdigit(s[pos]);
        if ((pos == 0 || prev_digit) && !cur_digit) {
            if (name_beg != -1 && value_beg != -1) {
                auto key = s.substr(name_beg, value_beg - name_beg);
                auto value = std::stoi(s.substr(value_beg, pos - value_beg));
                dims_.emplace(key, value);
            }
            name_beg = pos;
            value_beg = -1;
        }
        if (!prev_digit && cur_digit) { value_beg = pos; }
    }
}

int tile_config_t::dim(const std::string &dim_name) const {
    auto it = dims_.find(dim_name);
    if (it == dims_.end()) return 1;
    return it->second;
}

conv_config_params_t::conv_config_params_t(const std::string &s) {
    auto parts = split(s, " ");
    for (auto &part : parts) {
        auto sub_parts = split(part, "=");
        ir_assert(sub_parts.size() == 2);
        auto &name = sub_parts[0];
        auto &value = sub_parts[1];
        if (name == "s") {
            slm_ = slm_config_t(value);
        } else if (name == "T") {
            tg_tile_ = tile_config_t(value);
        } else if (name == "c") {
            check_slm_size_ = str_to_bool(value);
        } else {
            ir_error_not_expected() << part;
        }
    }
}

void conv_config_params_t::apply(conv_config_t &cfg) const {
    cfg.check_slm_size = check_slm_size_;
    cfg.slm_bufs = slm_.bufs;
    cfg.slm_sync_version = slm_.sync_version;
    cfg.gmem_bufs = slm_.gmem_bufs;

    auto &bh = cfg.bh;
    // Currently maybe_override_from_lookup_table() is called from two
    // places: during block setup (block helper is not frozen) and at the
    // very end to update the other parameters (when block helper is
    // frozen). Overriding is done in two steps because modifying block
    // helper blocks at the very end is not trivial due to dependencies.
    if (!bh->is_frozen()) {
        bh->set_tg_dim("oc", tg_tile_.dim("oc"));
        bh->set_tg_dim("mb", tg_tile_.dim("mb"));
        if (bh->has_dim("osp")) {
            bh->set_tg_dim("osp", tg_tile_.dim("osp"));
        } else {
            bh->set_tg_dim("ow", tg_tile_.dim("osp"));
        }
    }
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

conv_config_params_t conv_config_lookup_table_t::find(
        const conv_problem_t &prb, const hw_config_t &hw_cfg) const {
    auto key = prb.desc_str(/*print_mb=*/false);
    auto it = map_.find(key);
    if (it == map_.end()) return conv_config_params_t();
    for (auto &e : it->second) {
        if (e.filter.matches(prb, hw_cfg)) return e.params;
    }
    return conv_config_params_t();
}

void conv_config_lookup_table_t::add(const char *s_prb, const char *s_params) {
    conv_problem_filter_t filter(s_prb);
    conv_config_params_t params(s_params);
    map_[filter.key()].push_back(entry_t {filter, params});
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
