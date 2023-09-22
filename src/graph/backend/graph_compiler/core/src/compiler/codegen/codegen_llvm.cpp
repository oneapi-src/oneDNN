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
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include <compiler/codegen/llvm/shared_include.hpp>
// the visitor for lowering TIR to LLVM IR
#include <compiler/codegen/llvm/llvm_visitor.hpp>

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

#if SC_LLVM_BACKEND > 16
// starting from LLVM17, they use STL's optional container
template <typename T>
using Optional = std::optional<T>;
#endif

static std::string dump_module_to_string(Module *m) {
    std::string ret;
    raw_string_ostream os(ret);
    os << *m;
    return ret;
}

const_ir_module_ptr llvm_generator_pass::operator()(const_ir_module_ptr f) {
    auto passes = get_default_precodegen_passes(f->ctx_, gen_wrapper_);
    auto mod = run_precodegen_passes(passes, f);
    std::string unique_name;
    const auto &tmpdir = utils::compiler_configs_t::get_temp_dir_path();
    if (f->ctx_->flags_.debug_info_) {
        std::string file_name;
        file_name = "llvm_jit-" + utils::get_unique_name_for_file() + ".gcir";
        std::string unique_name = tmpdir + "/" + file_name;
        std::ofstream ofs;
        utils::open_file_for_write(ofs, unique_name);
        out_source_path_ = unique_name;
        print_ir_and_annotate_source_pos(*mod, ofs);
    } else {
        out_source_path_ = "";
    }

    codegen_llvm_vis_t vis {f->ctx_, llvm_ctx_, tmpdir, out_source_path_};
    auto timer = SC_SCOPED_TIMER_INFO("pass.time.llvm_generator_pass", "");
    for (auto &funct : mod->get_contents()) {
        vis.dispatch(funct);
    }
    if (f->ctx_->flags_.debug_info_) { vis.dbuilder_->finalize(); }
    out_module_ = std::move(vis.module_);
    SC_MODULE_INFO << dump_module_to_string(out_module_.get());
    return mod;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
