#include <stan/math/prim.hpp>
#include <test/unit/math/prim/functor/dde_test_functors.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using stan::test::dde::linear_lag_exact;
using stan::test::dde::linear_lag_rhs;
using stan::test::dde::scalar_lag_exact;
using stan::test::dde::scalar_lag_rhs;

std::vector<double> uniform_times(double start, double step, int n) {
  std::vector<double> ts(n);
  for (int i = 0; i < n; ++i) {
    ts[i] = start + step * (i + 1);
  }
  return ts;
}

Eigen::VectorXd vec(std::initializer_list<double> values) {
  Eigen::VectorXd v(values.size());
  int i = 0;
  for (double x : values) {
    v(i++) = x;
  }
  return v;
}

}  // namespace

// y' = -y(t - 1), y = 1 for t <= 0; the method of steps solution is
// piecewise polynomial. The output times include the breakpoints 1, 2, ...
TEST(dde_rk45_prim, single_delay_matches_method_of_steps) {
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({1.0});
  const std::vector<double> ts = uniform_times(0.0, 0.25, 24);

  auto ys = stan::math::dde_rk45_tol(scalar_lag_rhs(), y0, 0.0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr, 1.0);
  ASSERT_EQ(ys.size(), ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    ASSERT_EQ(ys[i].size(), 1);
    EXPECT_NEAR(ys[i](0), scalar_lag_exact(ts[i], 0.0, 1.0, 1.0, 1.0), 1e-8)
        << "t = " << ts[i];
  }
  // spot checks of the hand-derived pieces
  EXPECT_NEAR(ys[1](0), 0.5, 1e-8);          // 1 - t on [0, 1]
  EXPECT_NEAR(ys[3](0), 0.0, 1e-8);          // y(1) = 0
  EXPECT_NEAR(ys[7](0), -0.5, 1e-8);         // y(2) = -1/2
  EXPECT_NEAR(ys[11](0), -1.0 / 6.0, 1e-8);  // y(3) = -1/6
}

TEST(dde_rk45_prim, default_tolerances) {
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({1.0});
  const std::vector<double> ts = uniform_times(0.0, 0.5, 10);

  auto ys
      = stan::math::dde_rk45(scalar_lag_rhs(), y0, 0.0, ts, tau, nullptr, 1.0);
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(ys[i](0), scalar_lag_exact(ts[i], 0.0, 1.0, 1.0, 1.0), 1e-5);
  }
}

// nonzero t0, non-unit history, rate and delay; output times off the
// breakpoints
TEST(dde_rk45_prim, shifted_scaled_single_delay) {
  const double t0 = 2.5;
  const double y0_val = 0.7;
  const double a = 1.3;
  const Eigen::VectorXd y0 = vec({y0_val});
  const Eigen::VectorXd tau = vec({0.8});
  const std::vector<double> ts = uniform_times(t0, 0.37, 15);

  auto ys = stan::math::dde_rk45_tol(scalar_lag_rhs(), y0, t0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr, a);
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(ys[i](0), scalar_lag_exact(ts[i], t0, y0_val, a, 0.8), 1e-8)
        << "t = " << ts[i];
  }
}

// y' = -y(t - 1) - 0.5 y(t - 0.5), y = 1 for t <= 0
TEST(dde_rk45_prim, two_delays_match_method_of_steps) {
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({1.0, 0.5});
  Eigen::MatrixXd A1(1, 1);
  A1 << -1.0;
  Eigen::MatrixXd A2(1, 1);
  A2 << -0.5;
  const std::vector<Eigen::MatrixXd> A{A1, A2};
  const std::vector<double> ts = uniform_times(0.0, 0.3, 20);

  auto ys = stan::math::dde_rk45_tol(linear_lag_rhs(), y0, 0.0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr, A);
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(ys[i](0), linear_lag_exact(ts[i], 0.0, y0, 0.5, A, {2, 1})(0),
                1e-8)
        << "t = " << ts[i];
  }
}

// a coupled two-state system with different delays per coupling:
// y1' = -y2(t - 1) - 0.3 y1(t - 0.5), y2' = y1(t - 0.5)
TEST(dde_rk45_prim, coupled_system_matches_method_of_steps) {
  const Eigen::VectorXd y0 = vec({1.0, -0.5});
  const Eigen::VectorXd tau = vec({1.0, 0.5});
  Eigen::MatrixXd A1(2, 2);
  A1 << 0.0, -1.0, 0.0, 0.0;
  Eigen::MatrixXd A2(2, 2);
  A2 << -0.3, 0.0, 1.0, 0.0;
  const std::vector<Eigen::MatrixXd> A{A1, A2};
  const std::vector<double> ts = uniform_times(0.0, 0.4, 15);

  auto ys = stan::math::dde_rk45_tol(linear_lag_rhs(), y0, 0.0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr, A);
  for (size_t i = 0; i < ts.size(); ++i) {
    const Eigen::VectorXd exact
        = linear_lag_exact(ts[i], 0.0, y0, 0.5, A, {2, 1});
    ASSERT_EQ(ys[i].size(), 2);
    EXPECT_NEAR(ys[i](0), exact(0), 1e-8) << "t = " << ts[i];
    EXPECT_NEAR(ys[i](1), exact(1), 1e-8) << "t = " << ts[i];
  }
}

// a delay much smaller than the horizon caps the step size at the delay
TEST(dde_rk45_prim, small_delay) {
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({0.05});
  const std::vector<double> ts = uniform_times(0.0, 0.1, 10);

  auto ys = stan::math::dde_rk45_tol(scalar_lag_rhs(), y0, 0.0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr, 2.0);
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(ys[i](0), scalar_lag_exact(ts[i], 0.0, 1.0, 2.0, 0.05), 1e-8);
  }
}

TEST(dde_rk45_prim, y_lag_shape) {
  const Eigen::VectorXd y0 = vec({1.0, 2.0, 3.0});
  const Eigen::VectorXd tau = vec({1.0, 0.3});
  const std::vector<double> ts = uniform_times(0.0, 0.5, 4);

  EXPECT_NO_THROW(stan::math::dde_rk45(stan::test::dde::shape_check_rhs(), y0,
                                       0.0, ts, tau, nullptr, 2));
}

// without delays the DDE is an ODE: y' = -y
struct no_delay_rhs {
  template <typename T_t, typename T_y, typename T_lag>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
             std::ostream* msgs) const {
    if (y_lag.cols() != 0) {
      throw std::invalid_argument("expected no delays");
    }
    return -y;
  }
};

TEST(dde_rk45_prim, no_delays_is_an_ode) {
  const Eigen::VectorXd y0 = vec({1.0, 2.0});
  const Eigen::VectorXd tau(0);
  const std::vector<double> ts = uniform_times(0.0, 0.5, 6);

  auto ys = stan::math::dde_rk45_tol(no_delay_rhs(), y0, 0.0, ts, tau, 1e-10,
                                     1e-10, 100000, nullptr);
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(ys[i](0), std::exp(-ts[i]), 1e-9);
    EXPECT_NEAR(ys[i](1), 2.0 * std::exp(-ts[i]), 1e-9);
  }
}

TEST(dde_rk45_prim, argument_errors) {
  using stan::math::dde_rk45;
  using stan::math::dde_rk45_tol;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({1.0});
  const std::vector<double> ts = {0.5, 1.0, 2.0};

  EXPECT_NO_THROW(dde_rk45(f, y0, 0.0, ts, tau, nullptr, 1.0));

  EXPECT_THROW_MSG(dde_rk45(f, vec({nan}), 0.0, ts, tau, nullptr, 1.0),
                   std::domain_error, "initial state");
  EXPECT_THROW_MSG(dde_rk45(f, Eigen::VectorXd(0), 0.0, ts, tau, nullptr, 1.0),
                   std::invalid_argument, "initial state");
  EXPECT_THROW_MSG(dde_rk45(f, y0, inf, ts, tau, nullptr, 1.0),
                   std::domain_error, "initial time");
  EXPECT_THROW_MSG(
      dde_rk45(f, y0, 0.0, std::vector<double>{0.5, nan}, tau, nullptr, 1.0),
      std::domain_error, "times");
  EXPECT_THROW_MSG(
      dde_rk45(f, y0, 0.0, std::vector<double>{}, tau, nullptr, 1.0),
      std::invalid_argument, "times");
  // strictly increasing
  EXPECT_THROW_MSG(
      dde_rk45(f, y0, 0.0, std::vector<double>{0.5, 0.5}, tau, nullptr, 1.0),
      std::domain_error, "times");
  EXPECT_THROW_MSG(
      dde_rk45(f, y0, 0.0, std::vector<double>{1.0, 0.5}, tau, nullptr, 1.0),
      std::domain_error, "times");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 1.0, ts, tau, nullptr, 1.0),
                   std::domain_error, "initial time");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.5, ts, tau, nullptr, 1.0),
                   std::domain_error, "initial time");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, vec({0.0}), nullptr, 1.0),
                   std::domain_error, "delays");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, vec({-1.0}), nullptr, 1.0),
                   std::domain_error, "delays");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, vec({inf}), nullptr, 1.0),
                   std::domain_error, "delays");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, vec({nan}), nullptr, 1.0),
                   std::domain_error, "delays");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, tau, nullptr, nan),
                   std::domain_error, "dde parameters and data");
  EXPECT_THROW_MSG(dde_rk45(f, y0, 0.0, ts, tau, nullptr, inf),
                   std::domain_error, "dde parameters and data");

  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, 0.0, 1e-6, 1000, nullptr, 1.0),
      std::domain_error, "relative_tolerance");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, -1e-6, 1e-6, 1000, nullptr, 1.0),
      std::domain_error, "relative_tolerance");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, inf, 1e-6, 1000, nullptr, 1.0),
      std::domain_error, "relative_tolerance");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, 1e-6, 0.0, 1000, nullptr, 1.0),
      std::domain_error, "absolute_tolerance");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, 1e-6, nan, 1000, nullptr, 1.0),
      std::domain_error, "absolute_tolerance");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, 1e-6, 1e-6, 0, nullptr, 1.0),
      std::domain_error, "max_num_steps");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, y0, 0.0, ts, tau, 1e-6, 1e-6, -1, nullptr, 1.0),
      std::domain_error, "max_num_steps");

  // the function name is reported
  EXPECT_THROW_MSG(dde_rk45(f, vec({nan}), 0.0, ts, tau, nullptr, 1.0),
                   std::domain_error, "dde_rk45:");
  EXPECT_THROW_MSG(
      dde_rk45_tol(f, vec({nan}), 0.0, ts, tau, 1e-6, 1e-6, 10, nullptr, 1.0),
      std::domain_error, "dde_rk45_tol:");
}

TEST(dde_rk45_prim, integration_errors) {
  using stan::math::dde_rk45;
  using stan::math::dde_rk45_tol;
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau = vec({1.0});
  const std::vector<double> ts = {0.5, 1.0, 2.0};

  // the step size is at most the smallest delay, so this needs at least 1000
  // steps to reach the first output time
  EXPECT_THROW_MSG(
      dde_rk45_tol(scalar_lag_rhs(), y0, 0.0, std::vector<double>{10.0},
                   vec({0.01}), 1e-6, 1e-6, 100, nullptr, 1.0),
      std::domain_error, "max_num_steps");

  // a delay below the resolution of the integration times
  EXPECT_THROW_MSG(dde_rk45(scalar_lag_rhs(), y0, 1.0, std::vector<double>{2.0},
                            vec({1e-17}), nullptr, 1.0),
                   std::domain_error, "smallest delay");

  EXPECT_THROW_MSG(
      dde_rk45(stan::test::dde::wrong_size_rhs(), y0, 0.0, ts, tau, nullptr),
      std::invalid_argument, "dy_dt");

  EXPECT_THROW_MSG(
      dde_rk45(stan::test::dde::nan_rhs(), y0, 0.0, ts, tau, nullptr),
      std::domain_error, "step size became too small");

  EXPECT_THROW_MSG(
      dde_rk45(stan::test::dde::throwing_rhs(), y0, 0.0, ts, tau, nullptr),
      std::domain_error, "thrown by the right hand side");
}

// the solver keeps all its state per call, so concurrent solves give the
// same results as sequential ones
TEST(dde_rk45_prim, concurrent_solves) {
  const int num_threads = 8;
  const Eigen::VectorXd y0 = vec({1.0, -0.5});
  const Eigen::VectorXd tau = vec({1.0, 0.37});
  const std::vector<double> ts = uniform_times(0.0, 0.3, 20);
  auto solve = [&](int i) {
    const std::vector<double> theta{0.5 + 0.1 * i, 1.0, 0.8, 0.3 + 0.05 * i};
    return stan::math::dde_rk45_tol(stan::test::dde::nonlinear_rhs(), y0, 0.0,
                                    ts, tau, 1e-8, 1e-8, 100000, nullptr, theta,
                                    0.2);
  };

  std::vector<std::vector<Eigen::VectorXd>> sequential;
  for (int i = 0; i < num_threads; ++i) {
    sequential.push_back(solve(i));
  }

  std::vector<std::vector<Eigen::VectorXd>> concurrent(num_threads);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() { concurrent[i] = solve(i); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    ASSERT_EQ(concurrent[i].size(), ts.size());
    for (size_t k = 0; k < ts.size(); ++k) {
      EXPECT_EQ(concurrent[i][k](0), sequential[i][k](0));
      EXPECT_EQ(concurrent[i][k](1), sequential[i][k](1));
    }
  }
}
