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

#ifndef KERNEL_SELECTOR_HPP
#define KERNEL_SELECTOR_HPP

#include "gen_gemm_kernel_generator.hpp"

#include "kernel_catalog.hpp"
#include "kernel_evaluator.hpp"

#include <algorithm>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Basic kernel selection API.
struct MatchParams {
    MatchParams(const MatchParams &other) { *this = other; }
    MatchParams &operator=(const MatchParams &other) {
        selector = other.selector;
        sizes = other.sizes;
        precisionCExt = other.precisionCExt;
        ignoreSizes = other.ignoreSizes;
        stepping = other.stepping;
        alignment = other.alignment;
        unroll = other.unroll;
        temp = other.temp;

        copy_interned_values(other);
        return *this;
    }
    kcatalog::Selector selector;
    SizeParams sizes;
    char precisionCExt = 0;
    bool ignoreSizes = false;
    int stepping = 0;
    std::array<int, 3> alignment = {0, 0, 0};
    kcatalog::string tags, lateTags;
    std::array<int, 2> unroll = {0, 0};

    MatchParams() {}
    MatchParams(ngen::HW hw, const GEMMProblem &problem);

private:
    void copy_interned_values(const MatchParams &other) {
        auto interned = [&](const char *value) -> const char * {
            if (other.temp.data() <= value
                    && value < other.temp.data() + sizeof(temp)) {
                return temp.data() + (value - other.temp.data());
            }
            return value;
        };
        selector.precisions[0] = interned(other.selector.precisions[0]);
        selector.precisions[1] = interned(other.selector.precisions[1]);
        selector.precisions[2] = interned(other.selector.precisions[2]);
        selector.layouts[0] = interned(other.selector.layouts[0]);
        selector.layouts[1] = interned(other.selector.layouts[1]);
        selector.layouts[2] = interned(other.selector.layouts[2]);

        tags = interned(other.tags);
        lateTags = interned(other.lateTags);
    }
    std::array<char, 32> temp;
};

const kcatalog::Entry *select(const kcatalog::Catalog &catalog,
        const MatchParams &pattern, const EvaluateParams &eparams,
        EvaluateAuxOutput &aux);
const kcatalog::Entry *select(const kcatalog::Catalog &catalog, int npatterns,
        const MatchParams *patterns, const EvaluateParams &eparams,
        EvaluateAuxOutput &aux);

// Extended API for iterating over all matching kernels.
bool matches(const kcatalog::Entry &e, const MatchParams &pattern);

const kcatalog::Entry *lower_bound(
        const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);
const kcatalog::Entry *upper_bound(
        const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);

class EntryIterator {
public:
    EntryIterator(
            const kcatalog::Catalog &catalog_, const MatchParams &pattern_)
        : catalog(catalog_), pattern(pattern_) {
        begin = lower_bound(catalog_, pattern_.selector);
        end = upper_bound(catalog_, pattern_.selector);
        current = begin;
        findNextMatch();
    }

    operator bool() const { return current < end; }

    EntryIterator &operator++() {
        ++current;
        findNextMatch();
        return *this;
    }

    EntryIterator operator++(int) {
        auto old = *this;
        operator++();
        return old;
    }

    const kcatalog::Entry &operator*() const { return *current; }
    const kcatalog::Entry *operator->() const { return &*current; }

    friend bool operator==(const EntryIterator &i1, const EntryIterator &i2) {
        return (i1.current == i2.current);
    }
    friend bool operator!=(const EntryIterator &i1, const EntryIterator &i2) {
        return !(i1 == i2);
    }

protected:
    const kcatalog::Catalog &catalog;
    MatchParams pattern;
    const kcatalog::Entry *begin, *end, *current;

    void findNextMatch() {
        for (; current < end; current++) {
            if (matches(*current, pattern)) break;
        }
    }
};

inline EntryIterator match(
        const kcatalog::Catalog &catalog, const MatchParams &pattern) {
    return EntryIterator(catalog, pattern);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
