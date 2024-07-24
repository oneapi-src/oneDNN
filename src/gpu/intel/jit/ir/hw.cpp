/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/jit/ir/hw.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

std::vector<std::pair<ngen::HW, const char *>> hw_names = {
#define MAKE_PAIR(value) std::make_pair(ngen::HW::value, #value)
        MAKE_PAIR(Unknown),
        MAKE_PAIR(Gen9),
        MAKE_PAIR(Gen10),
        MAKE_PAIR(Gen11),
        MAKE_PAIR(XeLP),
        MAKE_PAIR(XeHP),
        MAKE_PAIR(XeHPG),
        MAKE_PAIR(XeHPC),
        MAKE_PAIR(Xe2),
#undef MAKE_PAIR
};

std::vector<std::pair<ngen::ProductFamily, const char *>> product_family_names
        = {
#define MAKE_PAIR(value, name) std::make_pair(ngen::ProductFamily::value, name)
                MAKE_PAIR(Unknown, "Unknown"),
                MAKE_PAIR(GenericGen9, "Gen9"),
                MAKE_PAIR(GenericGen10, "Gen10"),
                MAKE_PAIR(GenericGen11, "Gen11"),
                MAKE_PAIR(GenericXeLP, "XeLP"),
                MAKE_PAIR(GenericXeHP, "XeHP"),
                MAKE_PAIR(GenericXeHPG, "XeHPG"),
                MAKE_PAIR(DG2, "DG2"),
                MAKE_PAIR(MTL, "MTL"),
                MAKE_PAIR(ARL, "ARL"),
                MAKE_PAIR(GenericXeHPC, "XeHPC"),
                MAKE_PAIR(PVC, "PVC"),
                MAKE_PAIR(GenericXe2, "Xe2"),
#undef MAKE_PAIR
};

std::string to_string(ngen::HW hw) {
    for (auto &kv : hw_names) {
        if (hw == kv.first) return kv.second;
    }
    ir_error_not_expected() << static_cast<int>(hw);
    return {};
}

ngen::HW str_to_ngen_hw(const std::string &s) {
    for (auto &kv : hw_names) {
        if (utils::one_of(s, kv.second, ir_utils::to_lower(kv.second)))
            return kv.first;
    }
    ir_error_not_expected() << s;
    return ngen::HW::Unknown;
}

std::string to_string(ngen::ProductFamily family) {
    for (auto &kv : product_family_names) {
        if (family == kv.first) return kv.second;
    }
    ir_error_not_expected() << static_cast<int>(family);
    return {};
}

ngen::ProductFamily str_to_ngen_product_family(const std::string &s) {
    for (auto &kv : product_family_names) {
        if (utils::one_of(s, kv.second, ir_utils::to_lower(kv.second)))
            return kv.first;
    }
    ir_error_not_expected() << s;
    return ngen::ProductFamily::Unknown;
}

int hw_t::cache_line_size() const {
    switch (hw_) {
        case ngen::HW::Gen9:
        case ngen::HW::Gen10:
        case ngen::HW::Gen11:
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2: return 64;
        default: ir_error_not_expected();
    }
    return 0;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
