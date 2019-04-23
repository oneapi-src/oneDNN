/***************************************************************************
 *  Copyright (C) 2017 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  virtual_ptr.hpp
 *
 *  Description:
 *    Interface for SYCL buffers to behave as a non-dereferenceable pointer
 *
 * Authors:
 *
 *    Ruyman Reyes   Codeplay Software Ltd.
 *    Mehdi Goli     Codeplay Software Ltd.
 *    Vanya Yaneva   Codeplay Software Ltd.
 *
 **************************************************************************/

#ifndef CL_SYCL_SDK_CODEPLAY_VIRTUAL_PTR_HPP
#define CL_SYCL_SDK_CODEPLAY_VIRTUAL_PTR_HPP

#include <CL/sycl.hpp>

#include <cstddef>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace cl {
namespace sycl {
namespace codeplay {

using sycl_acc_target = cl::sycl::access::target;
using sycl_acc_mode = cl::sycl::access::mode;

/**
 * Default values for template arguments
 */
using buffer_data_type_t = uint8_t;
const sycl_acc_target default_acc_target = sycl_acc_target::global_buffer;
const sycl_acc_mode default_acc_mode = sycl_acc_mode::read_write;

/**
 * PointerMapper
 *  Associates fake pointers with buffers.
 *
 */
class PointerMapper {
 public:
  using base_ptr_t = std::uintptr_t;

  /* Structure of a virtual pointer
   *
   * |================================================|
   * |               POINTER ADDRESS                  |
   * |================================================|
   */
  struct virtual_pointer_t {
    /* Type for the pointers
     */
    base_ptr_t m_contents;

    /** Conversions from virtual_pointer_t to
     * void * should just reinterpret_cast the integer number
     */
    operator void*() const { return reinterpret_cast<void*>(m_contents); }

    /**
     * Convert back to the integer number.
     */
    operator base_ptr_t() const { return m_contents; }

    /**
     * Add a certain value to the pointer to create a
     * new pointer to that offset
     */
    virtual_pointer_t operator+(size_t off) { return m_contents + off; }

    /* Numerical order for sorting pointers in containers. */
    bool operator<(virtual_pointer_t rhs) const {
      return (static_cast<base_ptr_t>(m_contents) <
              static_cast<base_ptr_t>(rhs.m_contents));
    }

    bool operator>(virtual_pointer_t rhs) const {
      return (static_cast<base_ptr_t>(m_contents) >
              static_cast<base_ptr_t>(rhs.m_contents));
    }

    /**
     * Numerical order for sorting pointers in containers
     */
    bool operator==(virtual_pointer_t rhs) const {
      return (static_cast<base_ptr_t>(m_contents) ==
              static_cast<base_ptr_t>(rhs.m_contents));
    }

    /**
     * Simple forward to the equality overload.
     */
    bool operator!=(virtual_pointer_t rhs) const {
      return !(this->operator==(rhs));
    }

    /**
     * Converts a void * into a virtual pointer structure.
     * Note that this will only work if the void * was
     * already a virtual_pointer_t, but we have no way of
     * checking
     */
    virtual_pointer_t(const void* ptr)
        : m_contents(reinterpret_cast<base_ptr_t>(ptr)){};

    /**
     * Creates a virtual_pointer_t from the given integer
     * number
     */
    virtual_pointer_t(base_ptr_t u) : m_contents(u){};
  };

  /* Definition of a null pointer
   */
  const virtual_pointer_t null_virtual_ptr = nullptr;

  /**
   * Whether if a pointer is null or not.
   * A pointer is nullptr if the value is of null_virtual_ptr
   */
  static inline bool is_nullptr(virtual_pointer_t ptr) {
    return (static_cast<void*>(ptr) == nullptr);
  }

  /* basic type for all buffers
   */
  using buffer_t = cl::sycl::buffer<buffer_data_type_t>;

  /**
   * Node that stores information about a device allocation.
   * Nodes are sorted by size to organise a free list of nodes
   * that can be recovered.
   */
  struct pMapNode_t {
    buffer_t m_buffer;
    size_t m_size;
    bool m_free;

    pMapNode_t(buffer_t b, size_t size, bool f)
        : m_buffer{b}, m_size{size}, m_free{f} {
      m_buffer.set_final_data(nullptr);
    }

    bool operator<=(const pMapNode_t& rhs) { return (m_size <= rhs.m_size); }
  };

  /** Storage of the pointer / buffer tree
   */
  using pointerMap_t = std::map<virtual_pointer_t, pMapNode_t>;

  /**
   * Obtain the insertion point in the pointer map for
   * a pointer of the given size.
   * \param requiredSize Size attemted to reclaim
   */
  typename pointerMap_t::iterator get_insertion_point(size_t requiredSize) {
    typename pointerMap_t::iterator retVal;
    bool reuse = false;
    if (!m_freeList.empty()) {
      // try to re-use an existing block
      for (auto freeElem : m_freeList) {
        if (freeElem->second.m_size >= requiredSize) {
          retVal = freeElem;
          reuse = true;
          // Element is not going to be free anymore
          m_freeList.erase(freeElem);
          break;
        }
      }
    }
    if (!reuse) {
      retVal = std::prev(m_pointerMap.end());
    }
    return retVal;
  }

  /**
   * Returns an iterator to the node that stores the information
   * of the given virtual pointer from the given pointer map structure.
   * If pointer is not found, throws std::out_of_range.
   * If the pointer map structure is empty, throws std::out_of_range
   *
   * \param pMap the pointerMap_t structure storing all the pointers
   * \param virtual_pointer_ptr The virtual pointer to obtain the node of
   * \throws std::out:of_range if the pointer is not found or pMap is empty
   */
  typename pointerMap_t::iterator get_node(const virtual_pointer_t ptr) {
    if (this->count() == 0) {
      throw std::out_of_range("There are no pointers allocated");
    }
    if (is_nullptr(ptr)) {
      throw std::out_of_range("Cannot access null pointer");
    }
    // The previous element to the lower bound is the node that
    // holds this memory address
    auto node = m_pointerMap.lower_bound(ptr);
    // If the value of the pointer is not the one of the node
    // then we return the previous one
    if (node == m_pointerMap.end() || node->first != ptr) {
      if (node == std::begin(m_pointerMap)) {
        throw std::out_of_range("The pointer is not registered in the map");
      }
      --node;
    }

    return node;
  }

  bool is_vptr(const virtual_pointer_t ptr) const {
    if (this->count() == 0) {
      return false;
    }
    if (is_nullptr(ptr)) {
      return false;
    }
    // The previous element to the lower bound is the node that
    // holds this memory address
    auto node = m_pointerMap.lower_bound(ptr);
    // If the value of the pointer is not the one of the node
    // then we return the previous one
    if (node == m_pointerMap.end() || node->first != ptr) {
      if (node == std::begin(m_pointerMap)) {
        return false;
      }
      --node;
    }

    size_t off = (ptr - node->first);
    return off < node->second.m_size;
  }

  /* get_buffer.
   * Returns a buffer from the map using the pointer address
   */
  template <typename buffer_data_type = buffer_data_type_t>
  cl::sycl::buffer<buffer_data_type, 1> get_buffer(
      const virtual_pointer_t ptr) {
    //using sycl_buffer_t = cl::sycl::buffer<buffer_data_type, 1>;

    auto& map_node = get_node(ptr)->second;
    auto map_buffer = map_node.m_buffer;
    return map_buffer.reinterpret<buffer_data_type>(
        cl::sycl::range<1>{map_node.m_size / sizeof(buffer_data_type)});
  }

  /**
   * @brief Returns an accessor to the buffer of the given virtual pointer
   * @param accessMode
   * @param accessTarget
   * @param ptr The virtual pointer
   */
  template <sycl_acc_mode access_mode = default_acc_mode,
            sycl_acc_target access_target = default_acc_target,
            typename buffer_data_type = buffer_data_type_t>
  cl::sycl::accessor<buffer_data_type, 1, access_mode, access_target>
  get_access(const virtual_pointer_t ptr) {
    auto buf = get_buffer<buffer_data_type>(ptr);
    return buf.template get_access<access_mode>();
  }

  /**
   * @brief Returns an accessor to the buffer of the given virtual pointer
   *        in the given command group scope
   * @param accessMode
   * @param accessTarget
   * @param ptr The virtual pointer
   * @param cgh Reference to the command group scope
   */
  template <sycl_acc_mode access_mode = default_acc_mode,
            sycl_acc_target access_target = default_acc_target,
            typename buffer_data_type = buffer_data_type_t>
  cl::sycl::accessor<buffer_data_type, 1, access_mode, access_target>
  get_access(const virtual_pointer_t ptr, cl::sycl::handler& cgh) {
    auto buf = get_buffer<buffer_data_type>(ptr);
    return buf.template get_access<access_mode, access_target>(cgh);
  }

  /*
   * Returns the offset from the base address of this pointer.
   */
  inline std::ptrdiff_t get_offset(const virtual_pointer_t ptr) {
    // The previous element to the lower bound is the node that
    // holds this memory address
    return (ptr - get_node(ptr)->first);
  }

  /*
   * Returns the number of elements by which the given pointer is offset from
   * the base address.
   */
  template <typename buffer_data_type>
  inline size_t get_element_offset(const virtual_pointer_t ptr) {
    return get_offset(ptr) / sizeof(buffer_data_type);
  }

  /**
   * Constructs the PointerMapper structure.
   */
  PointerMapper(base_ptr_t baseAddress = 4096)
      : m_pointerMap{}, m_freeList{}, m_baseAddress{baseAddress} {
    if (m_baseAddress == 0) {
      throw std::invalid_argument(std::string("Base address cannot be zero"));
    }
  };

  /**
   * PointerMapper cannot be copied or moved
   */
  PointerMapper(const PointerMapper&) = delete;

  /**
   * Empty the pointer list
   */
  inline void clear() {
    m_freeList.clear();
    m_pointerMap.clear();
  }

  /* add_pointer.
   * Adds an existing pointer to the map and returns the virtual pointer id.
   */
  inline virtual_pointer_t add_pointer(const buffer_t& b) {
    return add_pointer_impl(b);
  }

  /* add_pointer.
   * Adds a pointer to the map and returns the virtual pointer id.
   */
  inline virtual_pointer_t add_pointer(buffer_t&& b) {
    return add_pointer_impl(b);
  }

  /**
   * @brief Fuses the given node with the previous nodes in the
   *        pointer map if they are free
   *
   * @param node A reference to the free node to be fused
   */
  void fuse_forward(typename pointerMap_t::iterator& node) {
    while (node != std::prev(m_pointerMap.end())) {
      // if following node is free
      // remove it and extend the current node with its size
      auto fwd_node = std::next(node);
      if (!fwd_node->second.m_free) {
        break;
      }
      auto fwd_size = fwd_node->second.m_size;
      m_freeList.erase(fwd_node);
      m_pointerMap.erase(fwd_node);

      node->second.m_size += fwd_size;
    }
  }

  /**
   * @brief Fuses the given node with the following nodes in the
   *        pointer map if they are free
   *
   * @param node A reference to the free node to be fused
   */
  void fuse_backward(typename pointerMap_t::iterator& node) {
    while (node != m_pointerMap.begin()) {
      // if previous node is free, extend it
      // with the size of the current one
      auto prev_node = std::prev(node);
      if (!prev_node->second.m_free) {
        break;
      }
      prev_node->second.m_size += node->second.m_size;

      // remove the current node
      m_freeList.erase(node);
      m_pointerMap.erase(node);

      // point to the previous node
      node = prev_node;
    }
  }

  /* remove_pointer.
   * Removes the given pointer from the map.
   * The pointer is allowed to be reused only if ReUse if true.
   */
  template <bool ReUse = true>
  void remove_pointer(const virtual_pointer_t ptr) {
    if (is_nullptr(ptr)) {
      return;
    }
    auto node = this->get_node(ptr);

    node->second.m_free = true;
    m_freeList.emplace(node);

    // Fuse the node
    // with free nodes before and after it
    fuse_forward(node);
    fuse_backward(node);

    // If after fusing the node is the last one
    // simply remove it (since it is free)
    if (node == std::prev(m_pointerMap.end())) {
      m_freeList.erase(node);
      m_pointerMap.erase(node);
    }
  }

  /* count.
   * Return the number of active pointers (i.e, pointers that
   * have been malloc but not freed).
   */
  size_t count() const { return (m_pointerMap.size() - m_freeList.size()); }

 private:
  /* add_pointer_impl.
   * Adds a pointer to the map and returns the virtual pointer id.
   * BufferT is either a const buffer_t& or a buffer_t&&.
   */
  template <class BufferT>
  virtual_pointer_t add_pointer_impl(BufferT b) {
    virtual_pointer_t retVal = nullptr;
    size_t bufSize = b.get_size() * sizeof(buffer_data_type_t);
    auto byte_buffer =
        b.template reinterpret<buffer_data_type_t>(cl::sycl::range<1>{bufSize});
    pMapNode_t p{byte_buffer, bufSize, false};

    // If this is the first pointer:
    if (m_pointerMap.empty()) {
      virtual_pointer_t initialVal{m_baseAddress};
      m_pointerMap.emplace(initialVal, p);
      return initialVal;
    }

    auto lastElemIter = get_insertion_point(bufSize);
    // We are recovering an existing free node
    if (lastElemIter->second.m_free) {
      lastElemIter->second.m_buffer = b;
      lastElemIter->second.m_free = false;

      // If the recovered node is bigger than the inserted one
      // add a new free node with the remaining space
      if (lastElemIter->second.m_size > bufSize) {
        // create a new node with the remaining space
        auto remainingSize = lastElemIter->second.m_size - bufSize;
        pMapNode_t p2{b, remainingSize, true};

        // update size of the current node
        lastElemIter->second.m_size = bufSize;

        // add the new free node
        auto newFreePtr = lastElemIter->first + bufSize;
        auto freeNode = m_pointerMap.emplace(newFreePtr, p2).first;
        m_freeList.emplace(freeNode);
      }

      retVal = lastElemIter->first;
    } else {
      size_t lastSize = lastElemIter->second.m_size;
      retVal = lastElemIter->first + lastSize;
      m_pointerMap.emplace(retVal, p);
    }
    return retVal;
  }

  /**
   * Compare two iterators to pointer map entries according to
   * the size of the allocation on the device.
   */
  struct SortBySize {
    bool operator()(typename pointerMap_t::iterator a,
                    typename pointerMap_t::iterator b) const {
      return ((a->first < b->first) && (a->second <= b->second)) ||
             ((a->first < b->first) && (b->second <= a->second));
    }
  };

  /* Maps the pointer addresses to buffer and size pairs.
   */
  pointerMap_t m_pointerMap;

  /* List of free nodes available for re-using
   */
  std::set<typename pointerMap_t::iterator, SortBySize> m_freeList;

  /* Base address used when issuing the first virtual pointer, allows users
   * to specify alignment. Cannot be zero. */
  size_t m_baseAddress;
};

/* remove_pointer.
 * Removes the given pointer from the map.
 * The pointer is allowed to be reused only if ReUse if true.
 */
template <>
inline void PointerMapper::remove_pointer<false>(const virtual_pointer_t ptr) {
  if (is_nullptr(ptr)) {
    return;
  }
  m_pointerMap.erase(this->get_node(ptr));
}

/**
 * Malloc-like interface to the pointer-mapper.
 * Given a size, creates a byte-typed buffer and returns a
 * fake pointer to keep track of it.
 * \param size Size in bytes of the desired allocation
 * \throw cl::sycl::exception if error while creating the buffer
 */
inline void* SYCLmalloc(size_t size, PointerMapper& pMap,
                        const property_list& pList = {}) {
  if (size == 0) {
    return nullptr;
  }
  // Create a generic buffer of the given size
  using sycl_buffer_t = cl::sycl::buffer<buffer_data_type_t, 1>;
  auto thePointer =
      pMap.add_pointer(sycl_buffer_t(cl::sycl::range<1>{size}, pList));
  // Store the buffer on the global list
  return static_cast<void*>(thePointer);
}

/**
 * Free-like interface to the pointer mapper.
 * Given a fake-pointer created with the virtual-pointer malloc,
 * destroys the buffer and remove it from the list.
 * If ReUse is false, the pointer is not added to the freeList,
 * it should be false only for sub-buffers.
 */
template <bool ReUse = true, typename PointerMapper>
inline void SYCLfree(void* ptr, PointerMapper& pMap) {
  pMap.template remove_pointer<ReUse>(ptr);
}

/**
 * Clear all the memory allocated by SYCL.
 */
template <typename PointerMapper>
inline void SYCLfreeAll(PointerMapper& pMap) {
  pMap.clear();
}

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl
#endif  // CL_SYCL_SDK_CODEPLAY_VIRTUAL_PTR_HPP
