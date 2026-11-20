#ifndef STAN_MATH_MEMORY_STACK_ALLOC_HPP
#define STAN_MATH_MEMORY_STACK_ALLOC_HPP

// TODO(Bob): <cstddef> replaces this ifdef in C++11, until then this
//            is best we can do to get safe pointer casts to uints.
#include <stdint.h>
#include <stan/math/prim/meta.hpp>
#include <cstdlib>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace stan {
namespace math {

/**
 * Return <code>true</code> if the specified pointer is aligned
 * on the number of bytes.
 *
 * This doesn't really make sense other than for powers of 2.
 *
 * @param ptr Pointer to test.
 * @param bytes_aligned Number of bytes of alignment required.
 * @return <code>true</code> if pointer is aligned.
 * @tparam Type of object to which pointer points.
 */
template <typename T>
STAN_STRONG_INLINE bool is_aligned(T* ptr, unsigned int bytes_aligned) {
  return (reinterpret_cast<uintptr_t>(ptr) % bytes_aligned) == 0U;
}

namespace internal {
const size_t DEFAULT_INITIAL_NBYTES = 1 << 16;  // 64KB

// FIXME: enforce alignment
// big fun to inline, but only called twice
STAN_STRONG_INLINE char* eight_byte_aligned_malloc(size_t size) {
  char* ptr = static_cast<char*>(malloc(size));
  if (unlikely(!ptr)) {
    return ptr;  // malloc failed to alloc
  } else if (unlikely(!is_aligned(ptr, 8U))) {
    std::stringstream s;
    s << "invalid alignment to 8 bytes, ptr="
      << reinterpret_cast<uintptr_t>(ptr) << std::endl;
    throw std::runtime_error(s.str());
  } else {
    return ptr;
  }
}

/**
 * Return the first address at or after `ptr` that is aligned to
 * `alignment` bytes.
 *
 * @param ptr Pointer to align.
 * @param alignment Alignment in bytes, a power of two.
 */
STAN_STRONG_INLINE char* align_up(char* ptr, size_t alignment) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t aligned
      = (address + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1);
  return ptr + (aligned - address);
}

/**
 * Hand the physical memory behind the whole pages inside
 * `[ptr, ptr + bytes)` back to the operating system. The address range
 * stays valid, readable and writable, and its next use gets fresh
 * zero-filled pages, so the contents of these pages are lost. Bytes
 * outside these whole pages, such as a memory allocator's bookkeeping next
 * to an allocation, are not touched.
 *
 * On Linux the pages are discarded with `madvise(MADV_DONTNEED)`. On macOS
 * they are replaced by fresh anonymous pages with `mmap(MAP_FIXED)`:
 * `madvise(MADV_FREE_REUSABLE)`, the other way to discard pages there,
 * would need a matching `madvise(MADV_FREE_REUSE)` before the pages are
 * written again, or the process's memory footprint would leave them out.
 *
 * @param ptr Start of the range. The range must lie within one allocation
 * of private, anonymous memory, such as a block from `malloc()`, whose
 * contents are no longer needed.
 * @param bytes Length of the range in bytes.
 * @return `true` if the pages were handed back or the range holds no whole
 * page, `false` if this platform offers no way to hand pages back (only
 * Linux and macOS do) or the system call failed.
 */
inline bool release_whole_pages(void* ptr, size_t bytes) noexcept {
#if defined(__linux__) || defined(__APPLE__)
  static const int64_t page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const uintptr_t page = static_cast<uintptr_t>(page_size);
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t first = (start + page - 1) / page * page;
  const uintptr_t last = (start + bytes) / page * page;
  if (ptr == nullptr || last <= first) {
    return true;
  }
  void* pages = reinterpret_cast<void*>(first);
  const size_t length = last - first;
#if defined(__APPLE__)
  return mmap(pages, length, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0)
         == pages;
#else
  return madvise(pages, length, MADV_DONTNEED) == 0;
#endif
#else
  (void)ptr;
  (void)bytes;
  return false;
#endif
}
}  // namespace internal

/**
 * An instance of this class provides a memory pool through
 * which blocks of raw memory may be allocated and then collected
 * simultaneously.
 *
 * This class is useful in settings where large numbers of small
 * objects are allocated and then collected all at once.  This may
 * include objects whose destructors have no effect.
 *
 * Memory is allocated on a stack of blocks.  Each block allocated
 * is twice as large as the previous one.  The memory may be
 * recovered, with the blocks being reused, or all blocks may be
 * freed, resetting the stack of blocks to its original state.
 * Once recovered, the physical memory behind the blocks may also be
 * handed back to the operating system while the blocks are kept, see
 * release_pages().
 *
 * Every allocation is at least 8 byte aligned. Larger alignments can be
 * requested through the `alignment` argument of `alloc()`.
 */
class stack_alloc {
 private:
  std::vector<char*> blocks_;  // storage for blocks,
                               // may be bigger than cur_block_
  std::vector<size_t> sizes_;  // could store initial & shift for others
  size_t cur_block_;           // index into blocks_ for next alloc
  char* cur_block_end_;        // ptr to cur_block_ptr_ + sizes_[cur_block_]
  char* next_loc_;             // ptr to next available spot in cur
                               // block
  // next three for keeping track of nested allocations on top of stack:
  std::vector<size_t> nested_cur_blocks_;
  std::vector<char*> nested_next_locs_;
  std::vector<char*> nested_cur_block_ends_;

  /**
   * Moves us to the next block of memory, allocating that block
   * if necessary, and allocates len bytes of memory within that
   * block, aligned to `alignment` bytes.
   *
   * @param len Number of bytes to allocate.
   * @param alignment Alignment in bytes, a power of two.
   * @return A pointer to the allocated memory.
   */
  STAN_STRONG_INLINE char* move_to_next_block(size_t len, size_t alignment) {
    // Blocks are 8 byte aligned, so leave room to align the start.
    const size_t needed = alignment > 8 ? len + alignment - 8 : len;
    ++cur_block_;
    // Find the next block (if any) containing at least needed bytes.
    while ((cur_block_ < blocks_.size()) && (sizes_[cur_block_] < needed)) {
      ++cur_block_;
    }
    // Allocate a new block if necessary.
    if (unlikely(cur_block_ >= blocks_.size())) {
      // New block should be max(2*size of last block, needed) bytes.
      size_t newsize = sizes_.back() * 2;
      if (newsize < needed) {
        newsize = needed;
      }
      blocks_.push_back(internal::eight_byte_aligned_malloc(newsize));
      if (!blocks_.back()) {
        throw std::bad_alloc();
      }
      sizes_.push_back(newsize);
    }
    char* result = internal::align_up(blocks_[cur_block_], alignment);
    // Get the object's state back in order.
    next_loc_ = result + len;
    cur_block_end_ = blocks_[cur_block_] + sizes_[cur_block_];
    return result;
  }

 public:
  /**
   * Construct a resizable stack allocator initially holding the
   * specified number of bytes.
   *
   * @param initial_nbytes Initial number of bytes for the
   * allocator.  Defaults to <code>(1 << 16) = 64KB</code> initial bytes.
   * @throws std::runtime_error if the underlying malloc is not 8-byte
   * aligned.
   */
  explicit stack_alloc(size_t initial_nbytes = internal::DEFAULT_INITIAL_NBYTES)
      : blocks_(1, internal::eight_byte_aligned_malloc(initial_nbytes)),
        sizes_(1, initial_nbytes),
        cur_block_(0),
        cur_block_end_(blocks_[0] + initial_nbytes),
        next_loc_(blocks_[0]) {
    if (!blocks_[0]) {
      throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
    }
  }

  /**
   * Destroy this memory allocator.
   *
   * This is implemented as a no-op as there is no destruction
   * required.
   */
  ~stack_alloc() {
    // free ALL blocks
    for (auto& block : blocks_) {
      if (block) {
        free(block);
      }
    }
  }

  /**
   * Return a newly allocated block of memory of the appropriate
   * size managed by the stack allocator.
   *
   * The allocated pointer will be aligned to `alignment` bytes, and at
   * least 8 bytes. If the number of bytes requested is not a multiple of
   * 8, the reserved space will be padded up to the next multiple of 8.
   *
   * This function may call C++'s <code>malloc()</code> function,
   * with any exceptions percolated through this function.
   *
   * @param len Number of bytes to allocate.
   * @param alignment Alignment in bytes, a power of two.
   * @return A pointer to the allocated memory.
   */
  STAN_STRONG_INLINE void* alloc(size_t len, size_t alignment = 8) {
    size_t pad = len % 8 == 0 ? 0 : 8 - len % 8;

    // Typically, just return and increment the next location.
    char* result
        = alignment <= 8 ? next_loc_ : internal::align_up(next_loc_, alignment);
    next_loc_ = result + len + pad;
    // Occasionally, we have to switch blocks.
    if (unlikely(next_loc_ >= cur_block_end_)) {
      result = move_to_next_block(len + pad, alignment);
    }
    return reinterpret_cast<void*>(result);
  }

  /**
   * Allocate an array on the arena of the specified size to hold
   * values of the specified template parameter type.
   *
   * @tparam T type of entries in allocated array.
   * @param[in] n size of array to allocate.
   * @return new array allocated on the arena.
   */
  template <typename T>
  STAN_STRONG_INLINE T* alloc_array(size_t n) {
    return static_cast<T*>(alloc(n * sizeof(T)));
  }

  /**
   * Recover all the memory used by the stack allocator.  The stack
   * of memory blocks allocated so far will be available for further
   * allocations.  To free memory back to the system, use the
   * function free_all(), or release_pages() to keep the blocks.
   */
  inline void recover_all() {
    cur_block_ = 0;
    next_loc_ = blocks_[0];
    cur_block_end_ = next_loc_ + sizes_[0];
  }

  /**
   * Store current positions before doing nested operation so can
   * recover back to start.
   */
  inline void start_nested() {
    nested_cur_blocks_.push_back(cur_block_);
    nested_next_locs_.push_back(next_loc_);
    nested_cur_block_ends_.push_back(cur_block_end_);
  }

  /**
   * recover memory back to the last start_nested call.
   */
  inline void recover_nested() {
    if (unlikely(nested_cur_blocks_.empty())) {
      recover_all();
    }

    cur_block_ = nested_cur_blocks_.back();
    nested_cur_blocks_.pop_back();

    next_loc_ = nested_next_locs_.back();
    nested_next_locs_.pop_back();

    cur_block_end_ = nested_cur_block_ends_.back();
    nested_cur_block_ends_.pop_back();
  }

  /**
   * Free all memory used by the stack allocator other than the
   * initial block allocation back to the system.  Note:  the
   * destructor will free all memory.
   */
  inline void free_all() {
    // frees all BUT the first (index 0) block
    for (size_t i = 1; i < blocks_.size(); ++i) {
      if (blocks_[i]) {
        free(blocks_[i]);
      }
    }
    sizes_.resize(1);
    blocks_.resize(1);
    recover_all();
  }

  /**
   * Hand the physical memory behind all blocks back to the operating
   * system while keeping the blocks themselves.
   *
   * recover_all() keeps every block for the next allocations, so the
   * memory of the largest stack ever allocated stays charged to the
   * process. free_all() returns the blocks to `free()`, which need not
   * return them to the operating system. This function instead tells the
   * operating system that the whole pages inside every block hold nothing
   * worth keeping (see internal::release_whole_pages()). The blocks stay
   * allocated, and later allocations reuse them as before; the first
   * write to each page gets a fresh zero-filled page from the system.
   *
   * Nothing may be allocated: call recover_all() first, and do not call
   * it while a nested allocation started by start_nested() is open.
   *
   * @return `true` if the memory of every block was handed back, `false`
   * if this platform offers no way to do so (only Linux and macOS do) or a
   * system call failed.
   * @throw std::logic_error if memory is still allocated or a nested
   * allocation is open; nothing is changed then.
   */
  inline bool release_pages() {
    if (cur_block_ != 0 || next_loc_ != blocks_[0]
        || !nested_cur_blocks_.empty()) {
      throw std::logic_error(
          "stack_alloc::release_pages(): memory is still allocated;"
          " call recover_all() first");
    }
    bool released = true;
    for (size_t i = 0; i < blocks_.size(); ++i) {
      released
          = internal::release_whole_pages(blocks_[i], sizes_[i]) && released;
    }
    return released;
  }

  /**
   * Return number of bytes allocated to this instance by the heap.
   * This is not the same as the number of bytes allocated through
   * calls to memalloc_.  The latter number is not calculable
   * because space is wasted at the end of blocks if the next
   * alloc request doesn't fit.  (Perhaps we could trim down to
   * what is actually used?)
   *
   * @return number of bytes allocated to this instance
   */
  inline size_t bytes_allocated() const {
    size_t sum = 0;
    for (size_t i = 0; i <= cur_block_; ++i) {
      sum += sizes_[i];
    }
    return sum;
  }

  /**
   * Return the approximate number of bytes handed out through calls to
   * alloc(). Blocks before the current one are counted in full, so any
   * space wasted at the end of a block when moving to the next one is
   * included.
   *
   * @return approximate number of bytes used on this instance
   */
  inline size_t approx_bytes_used() const {
    size_t sum = 0;
    for (size_t i = 0; i < cur_block_; ++i) {
      sum += sizes_[i];
    }
    return sum + static_cast<size_t>(next_loc_ - blocks_[cur_block_]);
  }

  /**
   * Indicates whether the memory in the pointer
   * is in the stack.
   *
   * @param[in] ptr memory location
   * @return true if the pointer is in the stack,
   *    false otherwise.
   */
  inline bool in_stack(const void* ptr) const {
    for (size_t i = 0; i < cur_block_; ++i) {
      if (ptr >= blocks_[i] && ptr < blocks_[i] + sizes_[i]) {
        return true;
      }
    }
    if (ptr >= blocks_[cur_block_] && ptr < next_loc_) {
      return true;
    }
    return false;
  }
};

}  // namespace math
}  // namespace stan
#endif
