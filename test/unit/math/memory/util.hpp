#ifndef TEST_UNIT_MATH_MEMORY_UTIL_HPP
#define TEST_UNIT_MATH_MEMORY_UTIL_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace stan {
namespace math {
namespace test {

/**
 * Whether this platform can hand pages back to the operating system, that
 * is whether `stack_alloc::release_pages()` and `release_memory()` return
 * `true`.
 */
#if defined(__linux__) || defined(__APPLE__)
constexpr bool can_release_pages = true;
#else
constexpr bool can_release_pages = false;
#endif

#if defined(__linux__) || defined(__APPLE__)
/**
 * The first and one-past-last address of the whole pages inside
 * `[ptr, ptr + bytes)`.
 */
inline std::pair<uintptr_t, uintptr_t> whole_pages(const void* ptr,
                                                   size_t bytes) {
  const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t first = (start + page - 1) / page * page;
  const uintptr_t last = (start + bytes) / page * page;
  return {first, last > first ? last : first};
}

/**
 * Number of bytes in the whole pages inside `[ptr, ptr + bytes)`.
 */
inline size_t whole_page_bytes(const void* ptr, size_t bytes) {
  auto pages = whole_pages(ptr, bytes);
  return pages.second - pages.first;
}

/**
 * Number of bytes in the whole pages inside `[ptr, ptr + bytes)` that are
 * in physical memory now, or -1 if that cannot be measured.
 */
inline int64_t resident_bytes(const void* ptr, size_t bytes) {
  auto pages = whole_pages(ptr, bytes);
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t length = pages.second - pages.first;
  if (length == 0) {
    return 0;
  }
  std::vector<unsigned char> present(length / page);
#if defined(__APPLE__)
  char* status = reinterpret_cast<char*>(present.data());
#else
  unsigned char* status = present.data();
#endif
  if (mincore(reinterpret_cast<void*>(pages.first), length, status) != 0) {
    return -1;
  }
  int64_t count = 0;
  for (unsigned char p : present) {
    count += p & 1;
  }
  return count * static_cast<int64_t>(page);
}
#endif

/**
 * The physical memory charged to this process in bytes, or -1 if it
 * cannot be measured: the resident set size on Linux, the memory footprint
 * (what Activity Monitor shows) on macOS.
 */
inline int64_t process_memory_bytes() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  int64_t size = 0;
  int64_t resident = 0;
  if (!(statm >> size >> resident)) {
    return -1;
  }
  return resident * static_cast<int64_t>(sysconf(_SC_PAGESIZE));
#elif defined(__APPLE__)
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count)
      != KERN_SUCCESS) {
    return -1;
  }
  return static_cast<int64_t>(info.phys_footprint);
#else
  return -1;
#endif
}

}  // namespace test
}  // namespace math
}  // namespace stan
#endif
