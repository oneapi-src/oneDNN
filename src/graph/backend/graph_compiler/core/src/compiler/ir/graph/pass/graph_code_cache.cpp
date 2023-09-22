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

#include <algorithm>
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include "pass.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/hash_utils.hpp>
#include <util/variant.hpp>

SC_MODULE(graph.pass.code_cache);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct prehashed_graph_for_code_share_t {
    context_ptr ctx_;
    variant<sc_graph_t, std::reference_wrapper<const sc_graph_t>> g_;
    // the ids for the base tensors for each shared const tensor
    std::vector<size_t> base_tensor_id_;
    // the offset in the base tensors for each shared const tensor
    std::vector<size_t> offset_;
    size_t hash_;
    static bool attr_filter(const sc_op *op, const std::string &attr) {
        if (attr == op_attr_key::const_input_cache) { return false; }
        if (op->isa<constant_op_t>()) {
            if (attr == "values") { return false; }
        }
        static std::unordered_set<std::string> ignorable_keys {
                attr_keys::scales, attr_keys::zero_points,
                attr_keys::data_scales, attr_keys::data_zero_points,
                attr_keys::weight_scales, attr_keys::weight_zero_points};
        if (ignorable_keys.find(attr) != ignorable_keys.end()) { return false; }
        return true;
    }

    prehashed_graph_for_code_share_t(const context_ptr &ctx, sc_graph_t &&g,
            const std::vector<size_t> &base_tensor_id,
            const std::vector<size_t> &offset, size_t ghash)
        : ctx_(ctx)
        , g_(std::move(g))
        , base_tensor_id_(base_tensor_id)
        , offset_(offset) {
        hash_ = ghash;
        hash_combine(hash_, base_tensor_id_);
        hash_combine(hash_, offset_);
        hash_combine(hash_, ctx_);
    }

    prehashed_graph_for_code_share_t(const context_ptr &ctx,
            std::reference_wrapper<const sc_graph_t> g,
            const std::vector<size_t> &base_tensor_id,
            const std::vector<size_t> &offset, size_t ghash)
        : ctx_(ctx), g_(g), base_tensor_id_(base_tensor_id), offset_(offset) {
        hash_ = ghash;
        hash_combine(hash_, base_tensor_id_);
        hash_combine(hash_, offset_);
        hash_combine(hash_, ctx_);
    }
    bool operator==(const prehashed_graph_for_code_share_t &other) const {
        if (hash_ != other.hash_) { return false; }
        if (ctx_ != other.ctx_) { return false; }
        if (base_tensor_id_ != other.base_tensor_id_) { return false; }
        if (offset_ != other.offset_) { return false; }
        return compare_graph(g_.cast<const sc_graph_t &>(),
                g_.cast<const sc_graph_t &>(), {}, attr_filter);
    }
};

std::ostream &operator<<(
        std::ostream &os, const prehashed_graph_for_code_share_t &v) {
    os << "hash=" << v.hash_
       << ", base tsr=" << utils::print_vector(v.base_tensor_id_)
       << ", offsets=" << utils::print_vector(v.offset_);
    return os;
}

struct graph_shared_ptr_hasher_t {
    size_t operator()(const prehashed_graph_for_code_share_t &v) const {
        return v.hash_;
    }
};

struct graph_code_cache_handle;
struct graph_code_cache_manager;

struct graph_code_cache_value_t {
    std::weak_ptr<graph_code_cache_handle> v_;
    std::shared_ptr<bool> deletion_flag_;
};

using graph_code_cache_map
        = std::unordered_map<prehashed_graph_for_code_share_t,
                graph_code_cache_value_t, graph_shared_ptr_hasher_t>;

struct graph_code_cache_handle {
    std::weak_ptr<jit_module_code> code_;
    statics_table_t module_data_template_;
    graph_code_cache_map::iterator iter_;
    std::shared_ptr<bool> deletion_flag_;
    std::shared_ptr<graph_code_cache_manager> mgr_;
    graph_code_cache_handle(const std::weak_ptr<jit_module_code> &code,
            statics_table_t &&module_data,
            const std::shared_ptr<graph_code_cache_manager> &mgr)
        : code_(code)
        , module_data_template_(std::move(module_data))
        , deletion_flag_(std::make_shared<bool>(false))
        , mgr_(mgr) {}
    ~graph_code_cache_handle();
};

static const char *shared_const_handle_name = "__shared_const_handle";
statics_table_t prepare_static_table_for_cached_code(
        graph_code_cache_handle &v, const sc_graph_t &orig_graph) {
    statics_table_t ret = v.module_data_template_.copy();
    if (auto bases = orig_graph.attrs_.get_or_null<
                     std::vector<std::shared_ptr<runtime::const_cache_proxy>>>(
                "shared_const_bases")) {
        if (!bases->empty()) {
            void **phandles
                    = (void **)ret.get_or_null(shared_const_handle_name);
            COMPILE_ASSERT(phandles,
                    "Cannot find shared_const_handle_name in module data");
            for (size_t i = 0; i < bases->size(); i++) {
                phandles[i] = (*bases)[i]->is_lazy_
                        ? (*bases)[i].get()
                        : (*bases)[i]->get_buffer_if_not_lazy();
            }
            ret.shared_tensors_ = orig_graph.attrs_.get<
                    std::vector<std::shared_ptr<cached_const_graph_tensor>>>(
                    "shared_const_tensors");
        }
    }
    return ret;
}

static std::shared_ptr<graph_code_cache_manager> get_cache_mgr();

struct graph_code_cache_manager {
    std::mutex lock_;
    graph_code_cache_map map_;

    std::shared_ptr<jit_module_code> internal_query(
            const prehashed_graph_for_code_share_t &v) {
        auto itr = map_.find(v);
        if (itr != map_.end()) {
            // find the key in the map, and make sure the weakptr are alive. If
            // the weakptr expired, remove the key from the map
            auto ret = some_opt(itr->second.v_.lock())
                               .map([](const std::shared_ptr<
                                            graph_code_cache_handle> &v) {
                                   return v->code_.lock();
                               })
                               .get_or_else(nullptr);
            if (!ret) {
                // notify the dtor of graph_code_cache_handle that the key is
                // already removed
                *itr->second.deletion_flag_ = false;
                map_.erase(itr);
                return nullptr;
            }
            return ret;
        }
        return nullptr;
    }

    std::shared_ptr<jit_module_code> query(
            const prehashed_graph_for_code_share_t &v) {
        std::lock_guard<std::mutex> guard {lock_};
        return internal_query(v);
    }

    std::shared_ptr<graph_code_cache_handle> insert(
            prehashed_graph_for_code_share_t &&v, const jit_module &m) {
        std::lock_guard<std::mutex> guard {lock_};
        if (internal_query(v)) {
            SC_MODULE_INFO << "The graph is already in the cache";
            return nullptr;
        }

        auto value = std::make_shared<graph_code_cache_handle>(
                m.code_, m.globals_.copy(), get_cache_mgr());
        auto graph_iter = map_.insert(std::make_pair(std::move(v),
                graph_code_cache_value_t {value, value->deletion_flag_}));
        assert(graph_iter.second);
        value->iter_ = graph_iter.first;
        m.code_->graph_cache_handle_ = value;
        SC_MODULE_INFO << "Putting into code cache, "
                       << graph_iter.first->first;
        if (auto sc_stream_temp
                = ::dnnl::impl::graph::gc::runtime::get_info_logging_stream(
                        "graph.pass.verbose.code_cache")) {
            print_graph(graph_iter.first->first.g_.cast<const sc_graph_t &>(),
                    *sc_stream_temp.stream_, true, true);
        }
        return value;
    }

    void remove(graph_code_cache_handle &v) {
        std::lock_guard<std::mutex> guard {lock_};
        /*
        see why the use of deletion_flag_ in const_graph_tensor_cache::remove
         */
        bool already_deleted = *v.deletion_flag_;
        if (already_deleted) { return; }
        if (v.iter_ != map_.end()) map_.erase(v.iter_);
    }
};

graph_code_cache_handle::~graph_code_cache_handle() {
    mgr_->remove(*this);
}

static std::shared_ptr<graph_code_cache_manager> get_cache_mgr() {
    static std::shared_ptr<graph_code_cache_manager> ret
            = std::make_shared<graph_code_cache_manager>();
    return ret;
}

size_t query_cached_code_of_context(const context_ptr &ctx) {
    auto mgr = get_cache_mgr();
    size_t ret = 0;
    std::lock_guard<std::mutex> guard {mgr->lock_};
    for (auto &kv : mgr->map_) {
        if (kv.first.ctx_ == ctx) ret++;
    }
    return ret;
}

std::shared_ptr<graph_code_cache_handle> register_code_in_graph_cache(
        const jit_module &m,
        std::shared_ptr<prehashed_graph_for_code_share_t> &&key) {
    return get_cache_mgr()->insert(std::move(*key), m);
}

SC_INTERNAL_API void graph_code_cache(sc_graph_t &mgr, const context_ptr &ctx) {
    if (!ctx->flags_.const_share_) { return; }
    // collect the base tensors and offsets
    std::vector<std::shared_ptr<runtime::const_cache_proxy>> bases;
    std::vector<size_t> base_ids;
    std::vector<size_t> offsets;
    std::vector<std::shared_ptr<cached_const_graph_tensor>> shared_tsr;
    op_visitor_t::dfs_topology_sort(mgr.ops_.size())
            .visit_graph(mgr, [&](op_visitor_t *, const sc_op_ptr &op) {
                if (auto buffer
                        = op->attrs_.get_or_null<std::vector<
                                  std::shared_ptr<cached_const_graph_tensor>>>(
                                op_attr_key::const_input_cache)) {
                    for (auto &buf : *buffer) {
                        shared_tsr.emplace_back(buf);
                        auto itr = std::find(
                                bases.begin(), bases.end(), buf->buf_base_);
                        if (itr != bases.end()) {
                            base_ids.push_back(
                                    static_cast<size_t>(itr - bases.begin()));
                        } else {
                            base_ids.push_back(bases.size());
                            bases.emplace_back(buf->buf_base_);
                        }
                        offsets.push_back(buf->offset_);
                    }
                }
            });
    mgr.attrs_["shared_const_bases"] = bases;
    mgr.attrs_["shared_const_tensors"] = shared_tsr;
    if (!ctx->flags_.mixed_fusion_) { return; }
    auto cache_mgr = get_cache_mgr();
    auto ghash
            = mgr.hash_contents(prehashed_graph_for_code_share_t::attr_filter);
    prehashed_graph_for_code_share_t key {ctx,
            std::reference_wrapper<const sc_graph_t>(mgr), base_ids, offsets,
            ghash};

    if (auto cached = cache_mgr->query(key)) {
        mgr.attrs_["graph_code_cache"] = cached;
        SC_MODULE_INFO << "Found cached code for the graph: " << key;
        return;
    }
    auto new_graph = copy_graph(mgr);
    // cleanup attrs
    for (auto &op : new_graph.ops_) {
        auto &attrs = op->attrs_.as_map();
        for (auto itr = attrs.begin(); itr != attrs.end();) {
            if (utils::string_startswith(itr->first, "temp.")
                    || !prehashed_graph_for_code_share_t::attr_filter(
                            op.get(), itr->first)) {
                itr = attrs.erase(itr);
                continue;
            }
            ++itr;
        }
    }
    auto cache_key = std::make_shared<prehashed_graph_for_code_share_t>(
            ctx, std::move(new_graph), base_ids, offsets, ghash);
    mgr.attrs_["graph_code_cache_key"] = cache_key;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
