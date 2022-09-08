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
    kcatalog::Selector selector;
    SizeParams sizes;
    char precisionCExt = 0;
    bool ignoreSizes = false;
    int stepping = 0;
    int alignment[3] = {0, 0, 0};
    kcatalog::string tags, lateTags;
    int unroll[2] = {0, 0};

    MatchParams() {}
    MatchParams(ngen::HW hw, const GEMMProblem &problem);

private:
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
