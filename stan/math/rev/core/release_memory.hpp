#ifndef STAN_MATH_REV_CORE_RELEASE_MEMORY_HPP
#define STAN_MATH_REV_CORE_RELEASE_MEMORY_HPP

#include <stan/math/memory/stack_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/empty_nested.hpp>
#include <stdexcept>
#include <vector>

namespace stan {
namespace math {

namespace internal {
/**
 * Hand the physical memory behind the storage of an empty vector back to
 * the operating system, keeping the storage itself.
 *
 * @tparam T Type of the vector's elements.
 * @param v Vector without elements.
 * @return `true` if the memory was handed back (or the vector has no
 * storage), `false` if this platform offers no way to do so or a system
 * call failed.
 */
template <typename T>
inline bool release_capacity(std::vector<T>& v) noexcept {
  return v.capacity() == 0
         || release_whole_pages(v.data(), v.capacity() * sizeof(T));
}
}  // namespace internal

/**
 * Hand the memory that the calling thread's autodiff stack keeps for reuse
 * back to the operating system, while keeping it reserved for the stack.
 *
 * recover_memory() rewinds the stack after a gradient, but it keeps all of
 * the stack's memory for the next one: the blocks of its arena, each twice
 * as large as the one before, and the storage of its lists of variables.
 * A thread therefore holds the memory of the largest gradient it has ever
 * computed for as long as it lives. This function tells the operating
 * system that this memory holds nothing worth keeping, so the system can
 * take back its physical memory now (see stack_alloc::release_pages()).
 * Nothing is freed: the next gradient on this thread uses the same memory
 * again, and only the first write to each of its pages costs a page fault.
 *
 * With `STAN_THREADS` every thread has its own stack, including the
 * worker threads of `reduce_sum()` and `map_rect()`; this function acts on
 * the calling thread's stack only.
 *
 * @return `true` if the memory was handed back, `false` if this platform
 * offers no way to do so (only Linux and macOS do) or a system call failed.
 * @throw std::logic_error if anything is on the stack or a nested
 * autodiff scope is open (call recover_memory() first); nothing is changed
 * then.
 */
inline bool release_memory() {
  auto& stack = *ChainableStack::instance_;
  if (!empty_nested() || !stack.var_stack_.empty()
      || !stack.var_nochain_stack_.empty() || !stack.var_alloc_stack_.empty()
      || stack.memalloc_.approx_bytes_used() != 0) {
    throw std::logic_error(
        "release_memory(): the autodiff stack is in use;"
        " call recover_memory() first");
  }
  bool released = stack.memalloc_.release_pages();
  released = internal::release_capacity(stack.var_stack_) && released;
  released = internal::release_capacity(stack.var_nochain_stack_) && released;
  released = internal::release_capacity(stack.var_alloc_stack_) && released;
  return released;
}

}  // namespace math
}  // namespace stan
#endif
