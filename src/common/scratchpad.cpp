/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_thread.hpp"

#include "scratchpad.hpp"

namespace mkldnn {
namespace impl {

/* Allocating memory buffers on a page boundary to reduce TLB/page misses */
const size_t page_size = 2097152;

/*
  Implementation of the scratchpad_t interface that is compatible with
  a concurrent execution
*/
struct concurent_scratchpad_t : public scratchpad_t {
    concurent_scratchpad_t(size_t size) {
        size_ = size;
        scratchpad_ = (char *) malloc(size, page_size);
        assert(scratchpad_ != nullptr);
    }

    ~concurent_scratchpad_t() {
        free(scratchpad_);
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    char *scratchpad_;
    size_t size_;
};

/*
  Implementation of the scratchpad_t interface that uses a global
  scratchpad
*/

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
// MSVC only supports <= OMP 2.0. Declaring static members
// to be threadprivate is only supported in OMP >= 3.0.
// As such, we need to implement this ourselves here.
struct global_scratchpad_t : public scratchpad_t {
	global_scratchpad_t(size_t size) {
		int tid = omp_get_thread_num();
		if (size > tls_[tid].size_) {
			if (tls_[tid].scratchpad_ != nullptr) free(tls_[tid].scratchpad_);
			tls_[tid].size_ = size;
			tls_[tid].scratchpad_ = (char *)malloc(size, page_size);
			assert(scratchpad_ != nullptr);
		}
		tls_[tid].reference_count_++;
	}
	~global_scratchpad_t() {
		int tid = omp_get_thread_num();
		tls_[tid].reference_count_--;
		if (tls_[tid].reference_count_ == 0) {
			free(tls_[tid].scratchpad_);
			tls_[tid].scratchpad_ = nullptr;
			tls_[tid].size_ = 0;
		}
	}

	void* operator new(size_t i)
	{
		return _aligned_malloc(i, 64);
	}

	void operator delete(void* p)
	{
		_aligned_free(p);
	}

	virtual char *get() const {
		return tls_[omp_get_thread_num()].scratchpad_;
	}

private:
	struct omp_2_fallback_member_tls
	{
		static char *scratchpad_;
		static size_t size_;
		static unsigned int reference_count_;

		char padding_[64 - (sizeof(scratchpad_) + sizeof(size_) + sizeof(reference_count_))];
	};

	__declspec(align(64)) omp_2_fallback_member_tls tls_[MAX_THREAD];
};

char *global_scratchpad_t::omp_2_fallback_member_tls::scratchpad_ = nullptr;
size_t global_scratchpad_t::omp_2_fallback_member_tls::size_ = 0;
unsigned int global_scratchpad_t::omp_2_fallback_member_tls::reference_count_ = 0;
#else
struct global_scratchpad_t : public scratchpad_t {
	global_scratchpad_t(size_t size) {
		if (size > size_) {
			if (scratchpad_ != nullptr) free(scratchpad_);
			size_ = size;
			scratchpad_ = (char *)malloc(size, page_size);
			assert(scratchpad_ != nullptr);
		}
		reference_count_++;
	}

	~global_scratchpad_t() {
		reference_count_--;
		if (reference_count_ == 0) {
			free(scratchpad_);
			scratchpad_ = nullptr;
			size_ = 0;
		}
	}

	virtual char *get() const {
		return scratchpad_;
	}

private:
	static char *scratchpad_;
	static size_t size_;
	static unsigned int reference_count_;
#pragma omp threadprivate(scratchpad_, size_, reference_count_)
};

char *global_scratchpad_t::scratchpad_ = nullptr;
size_t global_scratchpad_t::size_ = 0;
unsigned int global_scratchpad_t::reference_count_ = 0;
#endif // #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)

/*
   Scratchpad creation routine
*/
scratchpad_t *create_scratchpad(size_t size) {
#ifndef MKLDNN_ENABLE_CONCURRENT_EXEC
    return new global_scratchpad_t(size);
#else
    return new concurent_scratchpad_t(size);
#endif
}

}
}
