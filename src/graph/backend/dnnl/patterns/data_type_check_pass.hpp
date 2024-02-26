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

#ifndef GRAPH_BACKEND_DNNL_PATTERNS_DATA_TYPE_CHECK_PASS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_DATA_TYPE_CHECK_PASS_HPP

#include "graph/backend/dnnl/platform.hpp"
#include "graph/backend/fake/pattern_utils.hpp"

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

/*!
 * \brief dtype_check_pass_t generates a pass for checking unimplemented data 
 *        type.
 */
class dtype_check_pass_t : public graph::pass::pass_base {
public:
    explicit dtype_check_pass_t(std::string pbackend, std::string pname,
            std::vector<data_type_t> dtypes)
        : graph::pass::pass_base(std::move(pbackend), std::move(pname))
        , dt_to_check_(std::move(dtypes)) {
        // data type check passes should be executed first, hence should
        // have the highest priority.
        set_priority(50.f);
    }

    // the criteria of pass execution
    impl::status_t run(graph_t &agraph) override {
        // check if current pattern pass can be run on current graph
        engine_kind_t graph_engine_kind = agraph.get_engine_kind();
        if (get_engine_kind() != engine_kind::any_engine
                && get_engine_kind() != graph_engine_kind)
            return impl::status::success;

        std::vector<data_type_t> unsupported_dt;
        for (const auto &dt : dt_to_check_) {
            bool has_dtype_support
                    = platform::get_dtype_support_status(graph_engine_kind, dt);
            if (!has_dtype_support) unsupported_dt.emplace_back(dt);
        }
        if (unsupported_dt.empty()) return impl::status::success;

        dnnl::impl::graph::fake_impl::pattern_utils_t fake_pu;
        std::vector<op_t *> matched_op_list;

        // NOTE(zhitao): Currenrly there is no special handling for patterns
        // which owns unsupported data type internally for older platforms
        // but of which the corresponding compiled partitions can be executed,
        // e.g. int8-bf16 patterns such as dequant->tc->matmul->tc->quant.

        for (const std::shared_ptr<op_t> &aop : agraph.get_ops()) {
            bool meet_dtype_to_check {false};
            for (size_t i = 0; i < aop->num_inputs(); ++i) {
                const logical_tensor_t &oport
                        = aop->get_input_value(i)->get_logical_tensor();
                if (std::any_of(unsupported_dt.begin(), unsupported_dt.end(),
                            [&oport](data_type_t dt) {
                                return dt == oport.data_type;
                            })) {
                    meet_dtype_to_check = true;
                    break;
                }
            }
            for (size_t i = 0; i < aop->num_outputs(); ++i) {
                const logical_tensor_t &oport
                        = aop->get_output_value(i)->get_logical_tensor();
                if (std::any_of(unsupported_dt.begin(), unsupported_dt.end(),
                            [&oport](data_type_t dt) {
                                return dt == oport.data_type;
                            })) {
                    meet_dtype_to_check = true;
                    break;
                }
            }
            if (meet_dtype_to_check) matched_op_list.emplace_back(aop.get());
        }
        if (!matched_op_list.empty()) fake_pu.fuse(agraph, matched_op_list);

        return impl::status::success;
    }

private:
    std::vector<data_type_t> dt_to_check_;
};

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
