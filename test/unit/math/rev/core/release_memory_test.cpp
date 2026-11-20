#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/math/memory/util.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
/**
 * Value and gradient of a function of `n` variables, computed on the calling
 * thread's autodiff stack, which is recovered afterwards.
 */
std::vector<double> value_and_gradient(size_t n) {
  using stan::math::var;
  std::vector<var> x;
  for (size_t i = 0; i < n; ++i) {
    x.emplace_back(0.001 * i);
  }
  var lp = 0;
  for (size_t i = 0; i < n; ++i) {
    lp += stan::math::exp(stan::math::sin(x[i]) * x[i]);
  }
  lp.grad();
  std::vector<double> result{lp.val()};
  for (size_t i = 0; i < n; ++i) {
    result.push_back(x[i].adj());
  }
  stan::math::recover_memory();
  return result;
}
}  // namespace

TEST_F(AgradRev, release_memory_keeps_gradients) {
  const std::vector<double> before = value_and_gradient(100000);
  EXPECT_EQ(stan::math::test::can_release_pages, stan::math::release_memory());
  EXPECT_EQ(before, value_and_gradient(100000));
  // Releasing twice in a row is fine.
  EXPECT_EQ(stan::math::test::can_release_pages, stan::math::release_memory());
  EXPECT_EQ(stan::math::test::can_release_pages, stan::math::release_memory());
  EXPECT_EQ(before, value_and_gradient(100000));
}

TEST_F(AgradRev, release_memory_refuses_while_in_use) {
  using stan::math::ChainableStack;
  using stan::math::release_memory;
  using stan::math::var;
  {
    var x = 2.5;
    var y = x * x;
    EXPECT_THROW(release_memory(), std::logic_error);
    y.grad();
    EXPECT_EQ(6.25, y.val());
    EXPECT_EQ(5.0, x.adj());
  }
  stan::math::recover_memory();

  // A variable that is not chained.
  { var x(new stan::math::vari(1.5, false)); }
  EXPECT_THROW(release_memory(), std::logic_error);
  stan::math::recover_memory();

  // Memory on the arena without any variable.
  ChainableStack::instance_->memalloc_.alloc(16);
  EXPECT_THROW(release_memory(), std::logic_error);
  stan::math::recover_memory();

  // An open nested scope, even an empty one.
  stan::math::start_nested();
  EXPECT_THROW(release_memory(), std::logic_error);
  stan::math::recover_memory_nested();

  EXPECT_EQ(stan::math::test::can_release_pages, release_memory());
}

#if defined(__linux__) || defined(__APPLE__)
TEST_F(AgradRev, release_memory_hands_back_memory) {
  using stan::math::ChainableStack;
  using stan::math::test::process_memory_bytes;
  using stan::math::test::resident_bytes;
  const size_t n = 500000;

  // Size the stack, then hand its memory back: the baseline.
  value_and_gradient(n);
  EXPECT_TRUE(stan::math::release_memory());
  const int64_t baseline = process_memory_bytes();

  // The same gradient again: recover_memory() keeps all of its memory.
  value_and_gradient(n);
  const int64_t kept = process_memory_bytes();
  const auto& var_stack = ChainableStack::instance_->var_stack_;
  const size_t var_stack_bytes = var_stack.capacity() * sizeof(var_stack[0]);
  // The pages its list of variables has written stay in memory.
  const int64_t one_mib = 1 << 20;
  EXPECT_GT(resident_bytes(var_stack.data(), var_stack_bytes), one_mib);
  EXPECT_GT(kept - baseline, 64 * one_mib)
      << "kept: " << kept << ", baseline: " << baseline;

  EXPECT_TRUE(stan::math::release_memory());
  const int64_t released = process_memory_bytes();
  EXPECT_EQ(0, resident_bytes(var_stack.data(), var_stack_bytes));
  EXPECT_LT(released - baseline, (kept - baseline) / 10)
      << "kept: " << kept << ", released: " << released
      << ", baseline: " << baseline;
  RecordProperty("process_bytes_baseline", std::to_string(baseline));
  RecordProperty("process_bytes_kept", std::to_string(kept));
  RecordProperty("process_bytes_released", std::to_string(released));
}
#endif

#ifdef STAN_THREADS
TEST_F(AgradRev, release_memory_acts_on_calling_thread) {
  using stan::math::var;
  // The main thread's stack is in use throughout.
  var held = 3.0;
  std::vector<double> before;
  std::vector<double> after;
  bool released = false;
  std::thread worker([&]() {
    stan::math::ChainableStack thread_instance;
    before = value_and_gradient(10000);
    released = stan::math::release_memory();
    after = value_and_gradient(10000);
  });
  worker.join();
  EXPECT_EQ(stan::math::test::can_release_pages, released);
  EXPECT_EQ(before, after);
  EXPECT_THROW(stan::math::release_memory(), std::logic_error);
  EXPECT_EQ(3.0, held.val());
}
#endif
