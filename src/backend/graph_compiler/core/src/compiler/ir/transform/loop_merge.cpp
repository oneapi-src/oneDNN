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
#include "loop_merge.hpp"
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(loop_merger, SC_PASS_DEPENDS_ON(constant_folder),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

// if v is not stmts or v->seq_.size() > 1, return v
// if v is a stmts node with only one stmt, return the stmt
static stmt_c extract_single_stmt(stmt_c v) {
    auto vstmt = v.as<stmts_c>();
    if (vstmt.defined() && vstmt->seq_.size() == 1) {
        return vstmt->seq_.front();
    }
    return v;
}

class loop_merger_impl_t : public ir_visitor_t {
    bool merge_recursive = false;
    stmt_c visit(stmts_c v) override {
        if (v->seq_.size() <= 1) {
            // if there is only one or less statement, no chance to merge
            return ir_visitor_t::visit(std::move(v));
        }
        std::vector<stmt> newseq;
        newseq.reserve(v->seq_.size());
        bool changed = false;
        for (auto itr = v->seq_.begin(); itr != v->seq_.end(); ++itr) {
            auto news = dispatch(extract_single_stmt(*itr));
            if (!news.ptr_same(*itr)) { changed = true; }
            newseq.emplace_back(news.remove_const());
        }
        // the size of old seq
        int seqsize = newseq.size();
        // the size of seq after merging
        int newseqsize = 0;
        // the util function to push an stmt to newseq. Note that newseq has
        // already allocated enough size, we only need to write the value,
        // instead of pushing new ones
        auto repush_to_newseq = [&newseq, &newseqsize](stmt &&s) {
            newseq[newseqsize] = std::move(s);
            newseqsize++;
        };
        // the util function to move an stmt in newseq from src position to dst
        // position. We need to firstly insert it to dst position and remove it
        // in src position
        auto move_to_newseq = [&newseq, &newseqsize](stmt s, int src, int dst) {
            if (src == dst) {
                newseq[newseqsize] = std::move(s);
            } else {
                newseq.emplace(newseq.begin() + dst, std::move(s));
                newseq.erase(newseq.begin() + src + 1);
            }
            newseqsize++;
        };
        std::unordered_map<expr_c, expr> rmap;
        ir_copier_t ir_cper(rmap, /*create_var_tensor*/ false);

        for (int i = 0; i < seqsize; i++) {
            auto loop = newseq[i].as<for_loop_c>();
            // the index for the next iteration (will be added by 1)
            // it is used to skip the merged for-loops
            int nexti = i;

            if (loop.defined()) {
                // the editable copy of the loop
                for_loop writable_loop;
                if (merge_recursive
                        || (loop->attr_
                                && loop->attr_->has_key(
                                        stmt_attr_key::merge_loop))) {
                    // use this variable to record where should declare all the
                    // expressions of merged loop
                    int expr_insert_pos = newseqsize;
                    for (int j = i + 1; j < seqsize; j++) {
                        // for each next stmt, check if it is for_loop and is
                        // marked stmt_attr_key::merge_loop
                        auto nextloop = newseq[j].as<for_loop_c>();
                        if (nextloop.defined()
                                && (merge_recursive
                                        || (nextloop->attr_
                                                && nextloop->attr_->has_key(
                                                        stmt_attr_key::
                                                                merge_loop)))) {
                            if (!writable_loop.defined()) {
                                writable_loop
                                        = loop->remake().static_as<for_loop>();
                            }
                            int merged_loops = writable_loop->merge_all(stmt(),
                                    ir_cper(nextloop).static_as<for_loop>());
                            // if the next loop cannot be merged, break
                            if (merged_loops == 0) { break; }
                            // skip the merged loop in the for(int i...)
                            nexti++;
                        } else {
                            // if next stmt is var tensor def, move it up to
                            // expr_insert_pos
                            if (newseq[j].isa<define>()) {
                                move_to_newseq(std::move(newseq[j]), j,
                                        expr_insert_pos++);
                                // skip the merged loop in the for(int i...)
                                nexti++;
                                // set changed flag
                                changed = true;
                                continue;
                            } else {
                                // if we cannot merge the next stmt, break
                                break;
                            }
                        }
                    }
                }
                if (writable_loop.defined()) {
                    // if the loop has been merged with next loops
                    changed = true;
                    bool old_merge_recursive = merge_recursive;
                    merge_recursive = true;
                    writable_loop->body_
                            = dispatch(writable_loop->body_).remove_const();
                    merge_recursive = old_merge_recursive;
                    repush_to_newseq(std::move(writable_loop));
                } else {
                    repush_to_newseq(loop.remove_const());
                }
            } else {
                repush_to_newseq(std::move(newseq[i]));
            }
            i = nexti;
        }
        if (changed) {
            newseq.resize(newseqsize);
            return make_stmt<stmts_node_t>(std::move(newseq));
        }
        return v;
    }
};

func_c loop_merger_t::operator()(func_c f) {
    loop_merger_impl_t impl;
    return impl.dispatch(f);
};

expr_c loop_merger_t::operator()(expr_c f) {
    loop_merger_impl_t impl;
    return impl.dispatch(std::move(f));
};

stmt_c loop_merger_t::operator()(stmt_c f) {
    loop_merger_impl_t impl;
    return impl.dispatch(std::move(f));
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
