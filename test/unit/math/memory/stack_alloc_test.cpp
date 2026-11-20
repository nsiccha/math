#include <gtest/gtest.h>
#include <stan/math/memory/stack_alloc.hpp>
#include <test/unit/math/memory/util.hpp>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

TEST(MemoryStackAlloc, allocArray) {
  // just an example to show how alloc_array is used
  stan::math::stack_alloc allocator;
  double* x = allocator.alloc_array<double>(10U);
  for (int i = 0; i < 10; ++i)
    x[i] = 3.0;
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(3.0, x[i]);
}

struct biggy {
  double r[10];
};

TEST(MemoryStackAlloc, allocArrayBigger) {
  size_t N = 1000;
  size_t K = 10;
  stan::math::stack_alloc allocator;
  biggy* x = allocator.alloc_array<biggy>(N);
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < K; ++k)
      x[i].r[k] = k * i;
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < K; ++k)
      EXPECT_FLOAT_EQ(k * i, x[i].r[k]);
}
TEST(stack_alloc, bytes_allocated) {
  stan::math::stack_alloc allocator;
  EXPECT_LE(0L, allocator.bytes_allocated());
  for (size_t n = 1; n <= 10000; ++n) {
    allocator.alloc(n);
    size_t bytes_requested = (n * (n + 1)) / 2;
    size_t bytes_allocated = allocator.bytes_allocated();
    EXPECT_TRUE(bytes_requested <= bytes_allocated)
        << "bytes_requested: " << bytes_requested << std::endl
        << "bytes_allocated: " << bytes_allocated;
    // 1 << 16 is initial allocation;  *3 is to account for slop at end
    EXPECT_TRUE(bytes_allocated < ((1 << 16) + bytes_requested * 3));
  }
}

TEST(stack_alloc, is_aligned) {
  char* ptr = static_cast<char*>(malloc(1024));
  EXPECT_TRUE(stan::math::is_aligned(ptr, 1U));
  EXPECT_TRUE(stan::math::is_aligned(ptr, 2U));
  EXPECT_TRUE(stan::math::is_aligned(ptr, 4U));
  EXPECT_TRUE(stan::math::is_aligned(ptr, 8U));

  EXPECT_FALSE(stan::math::is_aligned(ptr + 1, 8U));
  // not very safe, but just a test
  free(ptr);
}

TEST(stack_alloc, alloc) {
  std::vector<double*> ds;
  std::vector<int*> is;
  std::vector<char*> cs;

  stan::math::stack_alloc allocator;

  for (int i = 0; i < 100000; ++i) {
    allocator.alloc(1317);
    double* foo = reinterpret_cast<double*>(allocator.alloc(sizeof(double)));
    *foo = 9.0;
    ds.push_back(foo);
    int* bar = reinterpret_cast<int*>(allocator.alloc(sizeof(int)));
    *bar = 17;
    is.push_back(bar);
    char* baz = reinterpret_cast<char*>(allocator.alloc(sizeof(char)));
    *baz = 3;
    cs.push_back(baz);
    allocator.alloc(13);

    EXPECT_FLOAT_EQ(9.0, *foo);
    EXPECT_EQ(17, *bar);
    EXPECT_EQ(3, *baz);
  }
  for (int i = 0; i < 10000; ++i) {
    EXPECT_FLOAT_EQ(9.0, *ds[i]);
    EXPECT_EQ(17, *is[i]);
    EXPECT_EQ(3, *cs[i]);
  }

  allocator.recover_all();
}

TEST(stack_alloc, alloc_aligned) {
  stan::math::stack_alloc allocator;
  int* x = allocator.alloc_array<int>(3);

  double* y = allocator.alloc_array<double>(4);
  EXPECT_TRUE(stan::math::is_aligned(y, 8));
  allocator.recover_all();
}

TEST(stack_alloc, alloc_over_aligned) {
  stan::math::stack_alloc allocator;
  // leave the next free byte 8 but not 64 byte aligned
  allocator.alloc(8);
  for (size_t alignment : {16U, 32U, 64U}) {
    double* x
        = static_cast<double*>(allocator.alloc(4 * sizeof(double), alignment));
    EXPECT_TRUE(stan::math::is_aligned(x, alignment)) << alignment;
    for (int i = 0; i < 4; ++i) {
      x[i] = i;
    }
    EXPECT_FLOAT_EQ(3.0, x[3]);
    // plain allocations after an aligned one stay 8 byte aligned
    EXPECT_TRUE(stan::math::is_aligned(allocator.alloc(sizeof(double)), 8U));
  }
}

TEST(stack_alloc, alloc_over_aligned_new_blocks) {
  stan::math::stack_alloc allocator;
  // larger than the initial block, so every allocation needs a new block
  const size_t big = 1 << 17;
  for (int i = 0; i < 4; ++i) {
    allocator.alloc(8);
    char* x = static_cast<char*>(allocator.alloc(big, 64));
    EXPECT_TRUE(stan::math::is_aligned(x, 64U));
    x[0] = 1;
    x[big - 1] = 2;
    EXPECT_TRUE(allocator.in_stack(x));
  }
  // reused blocks after recovering memory are aligned too
  allocator.recover_all();
  for (int i = 0; i < 4; ++i) {
    allocator.alloc(8);
    EXPECT_TRUE(stan::math::is_aligned(allocator.alloc(big, 64), 64U));
  }
}

TEST(stack_alloc, in_stack) {
  stan::math::stack_alloc allocator;

  double* x = allocator.alloc_array<double>(10U);
  double y = 0;

  EXPECT_TRUE(allocator.in_stack(x));
  EXPECT_FALSE(allocator.in_stack(&y));
}

TEST(stack_alloc, in_stack_second_block) {
  stan::math::stack_alloc allocator;

  char* x = allocator.alloc_array<char>(
      stan::math::internal::DEFAULT_INITIAL_NBYTES);
  EXPECT_TRUE(allocator.in_stack(x));
  EXPECT_FALSE(
      allocator.in_stack(x + stan::math::internal::DEFAULT_INITIAL_NBYTES));

  char* y = allocator.alloc_array<char>(1);
  EXPECT_TRUE(allocator.in_stack(x));
  EXPECT_TRUE(allocator.in_stack(y));
  // effect of padding
  EXPECT_TRUE(allocator.in_stack(y + 1));
  EXPECT_TRUE(allocator.in_stack(y + 7));

  EXPECT_FALSE(allocator.in_stack(y + 8));

  allocator.recover_all();
  EXPECT_FALSE(allocator.in_stack(x));
  EXPECT_FALSE(allocator.in_stack(y));
}

TEST(stack_alloc, release_pages_keeps_blocks) {
  stan::math::stack_alloc allocator;
  std::vector<char*> first;
  for (size_t n = 1; n <= 20; ++n) {
    char* x = static_cast<char*>(allocator.alloc(n * 100000));
    std::memset(x, 1, n * 100000);
    first.push_back(x);
  }
  const size_t allocated = allocator.bytes_allocated();
  allocator.recover_all();
  EXPECT_EQ(stan::math::test::can_release_pages, allocator.release_pages());

  // The same allocations reuse the same blocks, which still work.
  for (size_t n = 1; n <= 20; ++n) {
    char* x = static_cast<char*>(allocator.alloc(n * 100000));
    EXPECT_EQ(first[n - 1], x);
    std::memset(x, static_cast<int>(n), n * 100000);
  }
  for (size_t n = 1; n <= 20; ++n) {
    EXPECT_EQ(static_cast<char>(n), first[n - 1][0]);
    EXPECT_EQ(static_cast<char>(n), first[n - 1][n * 100000 - 1]);
  }
  EXPECT_EQ(allocated, allocator.bytes_allocated());
}

TEST(stack_alloc, release_pages_refuses_while_in_use) {
  stan::math::stack_alloc allocator;
  double* x = allocator.alloc_array<double>(4);
  x[3] = 4.5;
  EXPECT_THROW(allocator.release_pages(), std::logic_error);
  EXPECT_EQ(4.5, x[3]);
  EXPECT_TRUE(allocator.in_stack(x));

  allocator.recover_all();
  allocator.start_nested();
  EXPECT_THROW(allocator.release_pages(), std::logic_error);
  allocator.recover_nested();
  EXPECT_EQ(stan::math::test::can_release_pages, allocator.release_pages());
}

#if defined(__linux__) || defined(__APPLE__)
TEST(stack_alloc, release_pages_hands_back_memory) {
  using stan::math::test::process_memory_bytes;
  using stan::math::test::resident_bytes;
  using stan::math::test::whole_page_bytes;
  const size_t bytes = 64 << 20;  // 64 MiB, a block of its own
  stan::math::stack_alloc allocator;
  char* x = static_cast<char*>(allocator.alloc(bytes));
  std::memset(x, 1, bytes);
  const int64_t whole = whole_page_bytes(x, bytes);
  EXPECT_EQ(whole, resident_bytes(x, bytes));

  // recover_all() keeps the memory.
  allocator.recover_all();
  EXPECT_EQ(whole, resident_bytes(x, bytes));
  const int64_t kept = process_memory_bytes();

  EXPECT_TRUE(allocator.release_pages());
  EXPECT_EQ(0, resident_bytes(x, bytes));
  const int64_t released = process_memory_bytes();
  EXPECT_GT(kept - released, whole * 9 / 10)
      << "kept: " << kept << ", released: " << released;

  // The block is reused, and its memory is charged again once written.
  char* y = static_cast<char*>(allocator.alloc(bytes));
  EXPECT_EQ(x, y);
  std::memset(y, 2, bytes);
  EXPECT_EQ(whole, resident_bytes(y, bytes));
  const int64_t rewritten = process_memory_bytes();
  EXPECT_GT(rewritten - released, whole * 9 / 10);
  EXPECT_EQ(2, y[0]);
  EXPECT_EQ(2, y[bytes - 1]);
  RecordProperty("block_bytes", std::to_string(whole));
  RecordProperty("process_bytes_kept", std::to_string(kept));
  RecordProperty("process_bytes_released", std::to_string(released));
  RecordProperty("process_bytes_rewritten", std::to_string(rewritten));
}
#endif
