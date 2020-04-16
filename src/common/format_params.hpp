/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef FORMAT_PARAMS_HPP
#define FORMAT_PARAMS_HPP

#include "format_traits.hpp"
#include <unordered_map>

namespace mkldnn {
namespace impl {

/* provides format_traits data in runtime */
struct format_params_t {
    struct params_t {
        data_kind_t data_kind;
        block_format_t blk_fmt;
        int ndims;
        int ndims_sp;
        int blk_size;
    };
    static const params_t &get(memory_format_t fmt) {
        static const auto formats = get_formats();
        return formats.at(fmt);
    }

private:
    struct memory_format_t_hash {
        std::size_t operator()(memory_format_t fmt) const noexcept {
            return static_cast<size_t>(fmt);
        }
    };

    static const std::unordered_map<memory_format_t, format_params_t::params_t,
            format_params_t::memory_format_t_hash>
    get_formats() {
        std::unordered_map<memory_format_t, format_params_t::params_t,
                format_params_t::memory_format_t_hash>
                formats;
        populate_formats<(memory_format_t)0,
                memory_format_t::mkldnn_format_last>(formats);
        return formats;
    }

    template <memory_format_t B, memory_format_t E>
    static typename utils::enable_if<
            utils::is_defined<format_traits<B>>::value>::type
    populate_formats(std::unordered_map<memory_format_t,
            format_params_t::params_t, memory_format_t_hash> &formats) {
        formats.emplace(B,
                params_t{ format_traits<B>::data_kind,
                        format_traits<B>::blk_fmt, format_traits<B>::ndims,
                        format_traits<B>::ndims_sp,
                        format_traits<B>::blk_size });
        populate_formats<(memory_format_t)((int)B + 1), E>(formats);
    }
    template <memory_format_t B, memory_format_t E>
    static typename utils::enable_if<(
            B < E && !utils::is_defined<format_traits<B>>::value)>::type
    populate_formats(std::unordered_map<memory_format_t,
            format_params_t::params_t, memory_format_t_hash> &formats) {
        populate_formats<(memory_format_t)((int)B + 1), E>(formats);
    }
    template <memory_format_t B, memory_format_t E>
    static typename utils::enable_if<(
            B >= E && !utils::is_defined<format_traits<B>>::value)>::type
    populate_formats(std::unordered_map<memory_format_t,
            format_params_t::params_t, memory_format_t_hash> &) {}
};

} // namespace impl
} // namespace mkldnn

#endif