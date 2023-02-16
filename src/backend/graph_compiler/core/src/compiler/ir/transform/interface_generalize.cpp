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
#include <unordered_map>

#include <string>
#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../easy_build.hpp"
#include "interface_generalize.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(interface_generalizer,
        SC_PASS_DEPENDS_ON(dyn_tensor_transformer), SC_PASS_REQUIRE_STATE(),
        SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(),
        SC_PASS_UNSET_STATE());

const_ir_module_ptr interface_generalizer_t::operator()(
        const_ir_module_ptr in) {
    auto ret = in->copy();
    auto &funcs = ret->get_contents();
    auto len = funcs.size();
    builder::ir_builder_t builder;
    for (unsigned i = 0; i < len; i++) {
        auto f = funcs[i];
        if (f->body_.defined()
                && (!f->attr_
                        || !f->attr_->get_or_else(
                                function_attrs::private_, false))) {
            std::string wrapper_name = f->name_ + "_0wrapper";
            assert(!ret->get_func(wrapper_name));
            _function_(datatypes::void_t, wrapper_func,
                    _arg_("args", datatypes::generic,
                            {(int)f->params_.size()})) {
                _bind_(args);
                std::vector<expr> fargs;
                fargs.reserve(f->params_.size());
                for (uint64_t idx = 0; idx < f->params_.size(); idx++) {
                    auto &param = f->params_[idx];
                    assert(param->dtype_.lanes_ == 1);
                    if (param->dtype_ == datatypes::generic) {
                        fargs.emplace_back(args[idx]);
                    } else {
                        fargs.emplace_back(
                                builder::make_cast(param->dtype_, args[idx]));
                    }
                }
                builder.push_evaluate(builder::make_call(f->decl_, fargs));
            }
            wrapper_func->name_ = wrapper_name;
            wrapper_func->decl_->name_ = wrapper_name;
            if (f->attr_) {
                if (f->attr_->get_or_else(function_attrs::is_main, false)) {
                    wrapper_func->attr()[function_attrs::is_main] = true;
                }
                if (auto comments
                        = f->attr_->get_or_null<std::vector<std::string>>(
                                "comments")) {
                    std::vector<std::string> new_comments = {comments->at(0),
                            "@param args The array of arguments. It should "
                            "contain the following:"};
                    for (size_t i = 1; i < comments->size(); i++) {
                        auto &comment = (*comments)[i];
                        new_comments.emplace_back("  " + comment);
                        if (!comment.empty() && comment[0] == '@') {
                            new_comments.back()[2] = '-';
                        }
                    }
                    wrapper_func->attr()["comments"] = std::move(new_comments);
                }
            }
            ret->add_func({wrapper_func});
        }
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
