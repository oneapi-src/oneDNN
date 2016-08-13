/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef NSTL_HPP
#define NSTL_HPP

#include <vector>

#include "utils.hpp"

namespace mkldnn { namespace impl { namespace nstl {

template<typename T>
inline const T& max(const T& a, const T& b) {
    return a > b ? a : b;
}
template<typename T>
inline const T& min(const T& a, const T& b) {
    return a < b ? a : b;
}

// Rationale: MKL-DNN needs container implementations that do not generate
// dependencies on C++ run-time libraries.
//
// Implementation philosophy: caller is responsible to check if the operation
// is valid. The only functions that have to return status are those that
// depend on memory allocation or similar operations.
//
// This means that e.g. an operator [] does not have to check for boundaries.
// The caller should have checked the boundaries. If it did not we crash and
// burn: this is a bug in MKL-DNN and throwing an exception would not have been
// recoverable.
//
// On the other hand, insert() or resize() or a similar operation needs to
// return a status because the outcome depends on factors external to the
// caller. The situation is probably also not recoverable also, but MKL-DNN
// needs to be nice and report "out of memory" to the users.

enum nstl_status_t {
    success = 0,
    out_of_memory
};

template <typename T> class vector: public c_compatible {
private:
    std::vector<T> _impl;
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::size_type size_type;
    vector() {}
    ~vector() {}
    size_type size() const { return _impl.size(); }
    T& operator[] (size_type i) { return _impl[i]; }
    const T& operator[] (size_type i) const { return _impl[i]; }
    iterator begin() { return _impl.begin(); }
    iterator end() { return _impl.end(); }
    template <typename input_iterator>
    nstl_status_t insert(iterator pos, input_iterator begin, input_iterator end)
    {
        _impl.insert(pos, begin, end);
        return success;
    }
    void clear() { _impl.clear(); }
    void push_back(const T& t) { _impl.push_back(t); }
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
