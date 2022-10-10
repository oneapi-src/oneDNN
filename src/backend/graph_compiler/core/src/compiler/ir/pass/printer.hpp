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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_PRINTER_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_PRINTER_HPP

#include <ios>
#include <compiler/ir/viewer.hpp>

namespace sc {

class ir_module_t;
class ir_printer_t : public ir_viewer_t {
public:
    std::ostream &os_;
    int indents_ = 0;
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    ir_printer_t(std::ostream &os) : os_(os) {}

    std::ostream &do_dispatch(const expr_c &v);
    std::ostream &do_dispatch(const stmt_c &v);
    std::ostream &do_dispatch(const func_c &v);
    std::ostream &do_dispatch(const ir_module_t &v);

    func_c dispatch(func_c v) override;

#define SC_IR_PRINTER_METHODS_IMPL(node_type, ...) \
    void view(node_type##_c v) override;

    FOR_EACH_EXPR_IR_TYPE(SC_IR_PRINTER_METHODS_IMPL)
    FOR_EACH_STMT_IR_TYPE(SC_IR_PRINTER_METHODS_IMPL)
};

struct source_pos {
    int pos_;
    int line_;
    bool operator==(const source_pos &other) const {
        return pos_ == other.pos_ && line_ == other.line_;
    }
};

/**
 * @brief print the IR to the stream and annotate the source_pos on "source_pos"
 * attr of each IR node
 *
 * @param v the IR module
 * @param os the output stream
 */
void print_ir_and_annotate_source_pos(const ir_module_t &v, std::ostream &os);

} // namespace sc

#endif
