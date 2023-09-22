/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "dnnl_test_common.hpp"
#include "tests/test_thread.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using dim = memory::dim;
using dt = memory::data_type;
using tag = memory::format_tag;

enum class task_kind_t {
    conv_fwd,
};

class key_t {
public:
    key_t() = default;

    template <typename T>
    key_t(T t) {
        values_.push_back(static_cast<uint64_t>(t));
    }

    template <typename T, typename U>
    key_t(T t, U u) {
        values_.push_back(static_cast<uint64_t>(t));
        values_.push_back(static_cast<uint64_t>(u));
    }

    template <typename T>
    key_t append(T t) {
        auto ret = *this;
        ret.values_.push_back(static_cast<uint64_t>(t));
        return ret;
    }

    size_t get_hash() const {
        size_t ret = 0;
        for (auto v : values_) {
            ret ^= std::hash<uint64_t>()(v);
        }
        return ret;
    }

    bool operator==(const key_t &other) const {
        if (values_.size() != other.values_.size()) return false;
        for (size_t i = 0; i < values_.size(); i++) {
            if (values_[i] != other.values_[i]) return false;
        }
        return true;
    }

private:
    std::vector<uint64_t> values_;
};

struct key_hash_t {
    size_t operator()(const key_t &key) const { return key.get_hash(); }
};

struct key_equal_t {
    bool operator()(const key_t &a, const key_t &b) const { return a == b; }
};

template <typename T>
class resource_manager_t {
public:
    bool has(const key_t &key = 0) const { return cache_.count(key) != 0; }

    const T &get(const key_t &key = 0) const { return cache_.at(key); }

    void set(const key_t &key, const T &obj) {
        if (has(key)) return;
        cache_.emplace(key, obj);
    }

private:
    std::unordered_map<key_t, T, key_hash_t, key_equal_t> cache_;
};

class task_t {
public:
    virtual ~task_t() = default;
    virtual task_kind_t kind() const = 0;
    static std::mutex &mutex() { return mutex_; }

    virtual void create() = 0;
    virtual void execute() = 0;
    virtual void validate() = 0;

    void set_reuse_engine(bool value) { reuse_engine_ = value; }
    void set_reuse_stream(bool value) { reuse_stream_ = value; }
    void set_reuse_primitive(bool value) { reuse_primitive_ = value; }

    engine create_engine() const {
        key_t key;
        return create_object<engine>(reuse_engine_, key, engine_mgr_,
                [&] { return engine(get_test_engine_kind(), 0); });
    }

    stream create_stream(const engine &eng) const {
        key_t key(reinterpret_cast<uint64_t>(eng.get()));
        return create_object<stream>(
                reuse_stream_, key, stream_mgr_, [&] { return stream(eng); });
    }

    template <typename T>
    primitive create_primitive(const typename T::primitive_desc &pd) {
        key_t engine_key(reinterpret_cast<uint64_t>(pd.get_engine().get()));
        key_t key = engine_key.append(kind());
        return create_object<primitive>(
                reuse_primitive_, key, primitive_mgr_, [&] { return T(pd); });
    }

    memory create_memory(
            const memory::desc &d, const engine &eng, int value = 0) {
        auto ret = memory(d, eng);
        fill_memory(ret, value);
        return ret;
    }

protected:
    template <typename T, typename F>
    static T create_object(bool reuse, const key_t &key,
            resource_manager_t<T> &mgr, const F &func) {
        std::lock_guard<std::mutex> lock(mutex_);
        T ret;
        if (reuse && mgr.has(key)) {
            ret = mgr.get(key);
        } else {
            ret = func();
        }
        mgr.set(key, ret);
        return ret;
    }

    void fill_memory(const memory &mem, float value) {
        size_t sz = mem.get_desc().get_size();
        int elems = (int)(sz / sizeof(float));
        auto *ptr = mem.map_data<float>();
        GTEST_EXPECT_NE(ptr, nullptr);
        for (int i = 0; i < elems; i++) {
            ptr[i] = value;
        }
        mem.unmap_data(ptr);
    }

    static resource_manager_t<engine> engine_mgr_;
    static resource_manager_t<stream> stream_mgr_;
    static resource_manager_t<primitive> primitive_mgr_;
    static std::mutex mutex_;

    bool reuse_engine_ = false;
    bool reuse_stream_ = false;
    bool reuse_primitive_ = false;
};

resource_manager_t<engine> task_t::engine_mgr_;
resource_manager_t<stream> task_t::stream_mgr_;
resource_manager_t<primitive> task_t::primitive_mgr_;
std::mutex task_t::mutex_;

class conv_fwd_task_t : public task_t {
public:
    conv_fwd_task_t(int idx) : fill_value_(idx % 5) {}

    task_kind_t kind() const override { return task_kind_t::conv_fwd; }

    void create() override {
        eng_ = create_engine();

        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims wei_dims = {OC, IC, KH, KW};
        memory::dims dst_dims = {N, OC, OH, OW};
        memory::dims strides = {SH, SW};
        memory::dims padding_l = {PH, PW};
        memory::dims padding_r = {PH, PW};
        auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
        auto wei_md = memory::desc(wei_dims, dt::f32, tag::oihw);
        auto dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);

        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);

        pd_ = convolution_forward::primitive_desc(eng_,
                prop_kind::forward_training, algorithm::convolution_direct,
                src_md, wei_md, memory::desc(), dst_md, strides, padding_l,
                padding_r, attr);

        prim_ = create_primitive<convolution_forward>(pd_);

        size_t sz = pd_.scratchpad_desc().get_size();
        auto scratchpad = memory(memory::desc({(dim)sz}, dt::u8, tag::x), eng_);
        args_.emplace(DNNL_ARG_SCRATCHPAD, scratchpad);

        auto src_mem = create_memory(src_md, eng_, fill_value_);
        auto wei_mem = create_memory(wei_md, eng_, 1);
        auto dst_mem = create_memory(dst_md, eng_, 0);

        args_.emplace(DNNL_ARG_SRC, src_mem);
        args_.emplace(DNNL_ARG_WEIGHTS, wei_mem);
        args_.emplace(DNNL_ARG_DST, dst_mem);
    }

    void execute() override {
        auto strm = create_stream(eng_);
        prim_.execute(strm, args_);
        strm.wait();
    }

    void validate() override {
        auto &dst = args_.at(DNNL_ARG_DST);
        auto *ptr = dst.map_data<float>();
        GTEST_EXPECT_NE(ptr, nullptr);
        int elems = N * OC * OH * OW;
        bool ok = true;
        for (int i = 0; i < elems; i++) {
            if (ptr[i] != fill_value_ * IC * KH * KW) {
                ok = false;
                break;
            }
        }
        dst.unmap_data(ptr);
        EXPECT_TRUE(ok);
    }

private:
    static const dim N = 1;
    static const dim OC = 32;
    static const dim IC = 32;
    static const dim KH = 3, KW = 3;
    static const dim OH = 4, OW = 8;
    static const dim IH = OH + KH - 1, IW = OW + KW - 1;
    static const dim SH = 1, SW = 1;
    static const dim PH = 0, PW = 0;

    int fill_value_;
    engine eng_;
    convolution_forward::primitive_desc pd_;
    primitive prim_;
    std::unordered_map<int, memory> args_;
};

class test_concurrency_t : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _WIN32
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "GPU Windows is temporarily disabled due to long execution "
                "time.");
#endif
        SKIP_IF_CUDA(true, "Concurrent execution is not supported with CUDA.");

#ifdef DNNL_WITH_SYCL
        // XXX: Disable primitive cache to force creating new primitives each
        // time (and therefore new kernels) in different threads.
        // The reason for that is that there is a bug in SYCL that may cause
        // incorrect results of the primitive due to the same kernel being
        // submitted to different queues from from different threads.
        if (get_test_engine_kind() == engine::kind::gpu)
            set_primitive_cache_capacity(0);
#endif
        // This test doesn't work properly under SDE.
        const int len = 1024;
        char value_str[len];
        if (gtest_getenv("SDE_COMMAND_LINE", value_str, len) > 0)
            SKIP_IF(true, "Skipping concurrency test since executed under SDE");

        for (int i = 0; i < ntasks; i++) {
            auto task = std::make_shared<conv_fwd_task_t>(i);

            task->set_reuse_engine(i % 100 != 0);
            task->set_reuse_stream(i % 3 == 0);
            task->set_reuse_primitive(i % 5 == 0);

            catch_expected_failures(
                    [=]() { task->create(); }, false, dnnl_success);
            tasks_.emplace_back(task);
        }

        Test();
    }

    void Test() {
        dnnl::impl::parallel(nthreads, [this](int ithr, int nthr) {
            const int step = (ntasks + nthr - 1) / nthr;
            const int beg = ithr * step;
            const int end = std::min(beg + step, ntasks);
            for (int i = beg; i < end; i++)
                tasks_[i]->execute();
        });

        for (int i = 0; i < ntasks; i++) {
            tasks_[i]->validate();
        }
    }

    static const int ntasks;
    static const int nthreads;
    std::vector<std::shared_ptr<task_t>> tasks_;
    engine eng;
    stream strm;
};

const int test_concurrency_t::ntasks = 1000;
const int test_concurrency_t::nthreads = 100;

TEST_F(test_concurrency_t, Basic) {}

} // namespace dnnl
