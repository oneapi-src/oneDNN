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
#include "statics_table.hpp"
#include <iomanip>
#include <memory>
#include <string.h>
#include <runtime/context.hpp>
#include <runtime/runtime.hpp>
#include <util/assert.hpp>
#include <util/reflection.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static_data_t::static_data_t(const void *indata, size_t size)
    : aligned_buffer_t(size, runtime::get_default_stream()->engine_) {
    memcpy(data_, indata, size);
}

aligned_buffer_t::aligned_buffer_t(size_t size, runtime::engine_t *engine) {
    data_ = engine->vtable_->persistent_alloc(engine, size);
    size_ = size;
    engine_ = engine;
}

aligned_buffer_t::aligned_buffer_t(aligned_buffer_t &&other) {
    data_ = other.data_;
    size_ = other.size_;
    engine_ = other.engine_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.engine_ = nullptr;
}
aligned_buffer_t::~aligned_buffer_t() {
    if (data_) { engine_->vtable_->persistent_dealloc(engine_, data_); }
}

#define SC_CLASS_END2() \
    return &meta; \
    } \
    }

static int compare_data(void *lhs, void *rhs) {
    std::shared_ptr<static_data_t> *lv
            = reinterpret_cast<std::shared_ptr<static_data_t> *>(lhs);
    std::shared_ptr<static_data_t> *rv
            = reinterpret_cast<std::shared_ptr<static_data_t> *>(rhs);
    if (lv->get()->size_ < rv->get()->size_) { return -1; }
    if (lv->get()->size_ > rv->get()->size_) { return 1; }
    return memcmp(lv->get()->data_, rv->get()->data_, lv->get()->size_);
}

// clang-format off
SC_CLASS_WITH_NAME(shared_ptr_static_data, std::shared_ptr<static_data_t>)
    .get();
    meta.vtable_ = utils::make_unique<reflection::class_vtable_t>();
    meta.vtable_->compare_ = compare_data;
SC_CLASS_END2()
// clang-format on

void *statics_table_t::get(const std::string &name) const {
    auto itr = impl_.find(name);
    COMPILE_ASSERT(itr != impl_.end(),
            "Cannot find the name in globals table: " << name);
    return reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(data_.data_) + itr->second);
}

static constexpr size_t MAGIC = 0xc0ffeec011001010;
void statics_table_t::save_to_file(const std::string &path) const {
    COMPILE_ASSERT(initialized_size_ <= data_.size_, "Bad statics_table");
    FILE *ofs = fopen(path.c_str(), "wb");
    COMPILE_ASSERT(ofs, "Cannot open file for write: " << path);
    size_t data;
    data = MAGIC;
    fwrite(&data, sizeof(data), 1, ofs);
    data = data_.size_;
    fwrite(&data, sizeof(data), 1, ofs);
    data = initialized_size_;
    fwrite(&data, sizeof(data), 1, ofs);
    fwrite(data_.data_, initialized_size_, 1, ofs);
    fclose(ofs);
}

struct file_raii_t {
    FILE *f_;
    ~file_raii_t() {
        if (f_) fclose(f_);
    }
};
statics_table_t statics_table_t::load_from_file(const std::string &path) {
    FILE *ofs = fopen(path.c_str(), "rb");
    COMPILE_ASSERT(ofs, "Cannot open file for read: " << path);
    file_raii_t auto_close {ofs};
    size_t data = 0;
    size_t readitems;
    readitems = fread(&data, sizeof(data), 1, ofs);
    COMPILE_ASSERT(data == MAGIC, "Bad magic number");

    size_t total_size = 0;
    readitems = fread(&total_size, sizeof(total_size), 1, ofs);
    COMPILE_ASSERT(readitems == 1, "Bad EOF");

    size_t initialized_size = 0;
    readitems = fread(&initialized_size, sizeof(initialized_size), 1, ofs);
    COMPILE_ASSERT(readitems == 1, "Bad EOF");

    COMPILE_ASSERT(initialized_size <= total_size,
            "Expecting initialized_size <= total_size");
    static_data_t buf {total_size, runtime::get_default_stream()->engine_};

    if (initialized_size) {
        readitems = fread(buf.data_, initialized_size, 1, ofs);
        COMPILE_ASSERT(readitems == 1, "Bad EOF");
    }

    statics_table_t ret {std::move(buf)};
    ret.initialized_size_ = initialized_size;
    return ret;
}

void *statics_table_t::get_or_null(const std::string &name) const {
    auto itr = impl_.find(name);
    if (itr != impl_.end()) {
        return reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(data_.data_) + itr->second);
    }
    return nullptr;
}
void statics_table_t::add(const std::string &name, size_t d) {
    COMPILE_ASSERT(impl_.find(name) == impl_.end(),
            "Duplicated name in global tensors: " << name);
    impl_.insert(std::make_pair(name, d));
}

statics_table_t statics_table_t::copy() const {
    if (!data_.data_) { return statics_table_t(); }
    statics_table_t ret(aligned_buffer_t(data_.size_, data_.engine_));
    memcpy(ret.data_.data_, data_.data_, data_.size_);
    ret.impl_ = impl_;
    ret.initialized_size_ = initialized_size_;
    return ret;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
