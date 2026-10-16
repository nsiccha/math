#include <stan/math/rev.hpp>
#include <test/unit/math/prim/functor/dde_test_functors.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using stan::math::var;
using stan::test::dde::linear_lag_exact;
using stan::test::dde::linear_lag_rhs;
using stan::test::dde::nonlinear_rhs;
using stan::test::dde::scalar_lag_exact;
using stan::test::dde::scalar_lag_rhs;

using solution_d = std::vector<Eigen::VectorXd>;
using solution_v = std::vector<Eigen::Matrix<var, Eigen::Dynamic, 1>>;

std::vector<double> uniform_times(double start, double step, int n) {
  std::vector<double> ts(n);
  for (int i = 0; i < n; ++i) {
    ts[i] = start + step * (i + 1);
  }
  return ts;
}

/**
 * Check the gradients of every output of solve_v with respect to the
 * parameters against central finite differences of solve_d.
 */
void expect_gradients_match_finite_differences(
    const std::function<solution_v(const std::vector<var>&)>& solve_v,
    const std::function<solution_d(const std::vector<double>&)>& solve_d,
    const std::vector<double>& theta, double tolerance) {
  const size_t P = theta.size();
  std::vector<solution_d> fd(P);
  for (size_t p = 0; p < P; ++p) {
    const double eps = 1e-5 * std::max(1.0, std::fabs(theta[p]));
    std::vector<double> theta_plus = theta;
    std::vector<double> theta_minus = theta;
    theta_plus[p] += eps;
    theta_minus[p] -= eps;
    const solution_d y_plus = solve_d(theta_plus);
    const solution_d y_minus = solve_d(theta_minus);
    for (size_t i = 0; i < y_plus.size(); ++i) {
      fd[p].push_back((y_plus[i] - y_minus[i]) / (2 * eps));
    }
  }

  std::vector<var> theta_v(theta.begin(), theta.end());
  solution_v ys = solve_v(theta_v);
  const solution_d ys_d = solve_d(theta);
  ASSERT_EQ(ys.size(), ys_d.size());
  for (size_t i = 0; i < ys.size(); ++i) {
    for (Eigen::Index n = 0; n < ys[i].size(); ++n) {
      EXPECT_NEAR(ys[i](n).val(), ys_d[i](n), 1e-8);
      stan::math::set_zero_all_adjoints();
      ys[i](n).grad();
      for (size_t p = 0; p < P; ++p) {
        EXPECT_NEAR(theta_v[p].adj(), fd[p][i](n), tolerance)
            << "output " << i << ", component " << n << ", parameter " << p;
      }
    }
  }
  stan::math::recover_memory();
}

}  // namespace

/**
 * y' = -a y(t - tau), y = y0 for t <= t0, with any of y0, a and tau
 * autodiff variables; gradients against central differences of the exact
 * method of steps solution.
 */
template <typename T_y0, typename T_a, typename T_tau>
void expect_scalar_gradients_match_exact() {
  const double t0 = 0.5;
  const double y0_val = 0.7;
  const double a_val = 1.3;
  const double tau_val = 0.8;
  // output times stay away from the breakpoints t0 + k tau
  const std::vector<double> ts = uniform_times(t0, 0.37, 12);

  Eigen::Matrix<T_y0, Eigen::Dynamic, 1> y0(1);
  y0 << y0_val;
  T_a a = a_val;
  Eigen::Matrix<T_tau, Eigen::Dynamic, 1> tau(1);
  tau << tau_val;

  auto ys = stan::math::dde_rk45_tol(scalar_lag_rhs(), y0, t0, ts, tau, 1e-10,
                                     1e-10, 1000000, nullptr, a);
  ASSERT_EQ(ys.size(), ts.size());

  auto exact = [&](double y0_x, double a_x, double tau_x, double t) {
    return scalar_lag_exact(t, t0, y0_x, a_x, tau_x);
  };
  const double h = 1e-6;
  for (size_t i = 0; i < ts.size(); ++i) {
    const double t = ts[i];
    EXPECT_NEAR(stan::math::value_of(ys[i](0)),
                exact(y0_val, a_val, tau_val, t), 1e-8);
    const double d_y0 = (exact(y0_val + h, a_val, tau_val, t)
                         - exact(y0_val - h, a_val, tau_val, t))
                        / (2 * h);
    const double d_a = (exact(y0_val, a_val + h, tau_val, t)
                        - exact(y0_val, a_val - h, tau_val, t))
                       / (2 * h);
    const double d_tau = (exact(y0_val, a_val, tau_val + h, t)
                          - exact(y0_val, a_val, tau_val - h, t))
                         / (2 * h);
    stan::math::set_zero_all_adjoints();
    ys[i](0).grad();
    if constexpr (stan::is_var<T_y0>::value) {
      EXPECT_NEAR(y0(0).adj(), d_y0, 1e-7) << "d/dy0 at t = " << t;
    }
    if constexpr (stan::is_var<T_a>::value) {
      EXPECT_NEAR(a.adj(), d_a, 1e-7) << "d/da at t = " << t;
    }
    if constexpr (stan::is_var<T_tau>::value) {
      EXPECT_NEAR(tau(0).adj(), d_tau, 1e-7) << "d/dtau at t = " << t;
      // y(t) does not depend on tau before the first breakpoint
      if (t < t0 + tau_val) {
        EXPECT_EQ(tau(0).adj(), 0.0);
      }
    }
  }
  stan::math::recover_memory();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_all_var) {
  expect_scalar_gradients_match_exact<var, var, var>();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_y0_var) {
  expect_scalar_gradients_match_exact<var, double, double>();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_args_var) {
  expect_scalar_gradients_match_exact<double, var, double>();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_tau_var) {
  expect_scalar_gradients_match_exact<double, double, var>();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_y0_tau_var) {
  expect_scalar_gradients_match_exact<var, double, var>();
}

TEST_F(AgradRev, dde_rk45_scalar_gradients_args_tau_var) {
  expect_scalar_gradients_match_exact<double, var, var>();
}

// At the default tolerances the gradient with respect to the delay needs the
// first stage derivative after the breakpoint t0 + tau to be recomputed with
// the derivative of the solution after t0 (without it the error is about
// 1e-4 here instead of about 4e-6)
TEST_F(AgradRev, dde_rk45_tau_gradient_at_default_tolerances) {
  const double t0 = 0.5;
  const double y0_val = 0.7;
  const double a_val = 1.3;
  const double tau_val = 0.8;
  const std::vector<double> ts = uniform_times(t0, 0.37, 12);

  Eigen::VectorXd y0(1);
  y0 << y0_val;
  Eigen::Matrix<var, Eigen::Dynamic, 1> tau(1);
  tau << tau_val;
  auto ys
      = stan::math::dde_rk45(scalar_lag_rhs(), y0, t0, ts, tau, nullptr, a_val);
  const double h = 1e-6;
  for (size_t i = 0; i < ts.size(); ++i) {
    const double d_tau
        = (scalar_lag_exact(ts[i], t0, y0_val, a_val, tau_val + h)
           - scalar_lag_exact(ts[i], t0, y0_val, a_val, tau_val - h))
          / (2 * h);
    stan::math::set_zero_all_adjoints();
    ys[i](0).grad();
    EXPECT_NEAR(tau(0).adj(), d_tau, 2e-5) << "t = " << ts[i];
  }
  stan::math::recover_memory();
}

// y' = -a1 y(t - tau1) - a2 y(t - tau2) with every input an autodiff
// variable: the solver's gradient against finite differences of the solver
TEST_F(AgradRev, dde_rk45_two_delays_gradients_match_finite_differences) {
  const double t0 = 0.0;
  const std::vector<double> ts = uniform_times(t0, 0.45, 12);
  // theta = (y0, a1, a2, tau1, tau2)
  const std::vector<double> theta{1.2, 1.0, 0.5, 1.0, 0.45};

  auto solve = [&](const auto& th) {
    using T = std::decay_t<decltype(th[0])>;
    Eigen::Matrix<T, Eigen::Dynamic, 1> y0(1);
    y0 << th[0];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A1(1, 1);
    A1 << -th[1];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A2(1, 1);
    A2 << -th[2];
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> A{A1, A2};
    Eigen::Matrix<T, Eigen::Dynamic, 1> tau(2);
    tau << th[3], th[4];
    return stan::math::dde_rk45_tol(linear_lag_rhs(), y0, t0, ts, tau, 1e-12,
                                    1e-12, 1000000, nullptr, A);
  };
  expect_gradients_match_finite_differences(
      [&](const std::vector<var>& th) { return solve(th); },
      [&](const std::vector<double>& th) { return solve(th); }, theta, 1e-5);
}

// the nonlinear two-state system with every input an autodiff variable
// (the data argument c stays double)
TEST_F(AgradRev, dde_rk45_nonlinear_gradients_match_finite_differences) {
  const double t0 = 0.3;
  const std::vector<double> ts = uniform_times(t0, 0.6, 10);
  // theta = (y0_1, y0_2, k1, r, k2, k3, tau1, tau2)
  const std::vector<double> theta{1.0, 0.4, 0.7, 1.5, 0.9, 0.35, 1.1, 0.65};

  auto solve = [&](const auto& th) {
    using T = std::decay_t<decltype(th[0])>;
    Eigen::Matrix<T, Eigen::Dynamic, 1> y0(2);
    y0 << th[0], th[1];
    std::vector<T> rates{th[2], th[3], th[4], th[5]};
    Eigen::Matrix<T, Eigen::Dynamic, 1> tau(2);
    tau << th[6], th[7];
    return stan::math::dde_rk45_tol(nonlinear_rhs(), y0, t0, ts, tau, 1e-12,
                                    1e-12, 1000000, nullptr, rates, 0.2);
  };
  expect_gradients_match_finite_differences(
      [&](const std::vector<var>& th) { return solve(th); },
      [&](const std::vector<double>& th) { return solve(th); }, theta, 1e-5);
}

// only the delays are autodiff variables
TEST_F(AgradRev, dde_rk45_nonlinear_tau_gradients_match_finite_differences) {
  const double t0 = 0.0;
  const std::vector<double> ts = uniform_times(t0, 0.5, 10);
  const std::vector<double> theta{1.1, 0.65};

  auto solve = [&](const auto& th) {
    using T = std::decay_t<decltype(th[0])>;
    Eigen::VectorXd y0(2);
    y0 << 1.0, 0.4;
    std::vector<double> rates{0.7, 1.5, 0.9, 0.35};
    Eigen::Matrix<T, Eigen::Dynamic, 1> tau(2);
    tau << th[0], th[1];
    return stan::math::dde_rk45_tol(nonlinear_rhs(), y0, t0, ts, tau, 1e-12,
                                    1e-12, 1000000, nullptr, rates, 0.2);
  };
  expect_gradients_match_finite_differences(
      [&](const std::vector<var>& th) { return solve(th); },
      [&](const std::vector<double>& th) { return solve(th); }, theta, 1e-5);
}

// a failure inside the right hand side propagates and leaves the autodiff
// stack usable
TEST_F(AgradRev, dde_rk45_errors_propagate) {
  Eigen::Matrix<var, Eigen::Dynamic, 1> y0(1);
  y0 << 1.0;
  Eigen::Matrix<var, Eigen::Dynamic, 1> tau(1);
  tau << 1.0;
  const std::vector<double> ts{0.5, 1.0, 2.0};

  EXPECT_THROW_MSG(stan::math::dde_rk45(stan::test::dde::throwing_rhs(), y0,
                                        0.0, ts, tau, nullptr),
                   std::domain_error, "thrown by the right hand side");
  EXPECT_THROW_MSG(stan::math::dde_rk45(stan::test::dde::wrong_size_rhs(), y0,
                                        0.0, ts, tau, nullptr),
                   std::invalid_argument, "dy_dt");
  EXPECT_THROW_MSG(stan::math::dde_rk45(stan::test::dde::nan_rhs(), y0, 0.0, ts,
                                        tau, nullptr),
                   std::domain_error, "step size became too small");
  // no nested autodiff scope was left open by the failures
  EXPECT_TRUE(
      stan::math::ChainableStack::instance_->nested_var_stack_sizes_.empty());

  auto ys = stan::math::dde_rk45(scalar_lag_rhs(), y0, 0.0, ts, tau, nullptr,
                                 var(1.0));
  ys[2](0).grad();
  EXPECT_NEAR(ys[2](0).val(), -0.5, 1e-5);
  stan::math::recover_memory();
}

#ifdef STAN_THREADS
// several reverse mode solves running concurrently on their own autodiff
// stacks give the same values and gradients as sequential ones
TEST_F(AgradRev, dde_rk45_concurrent_solves) {
  const int num_threads = 8;
  const std::vector<double> ts = uniform_times(0.0, 0.3, 15);

  auto solve_and_grad = [&](int i) {
    Eigen::Matrix<var, Eigen::Dynamic, 1> y0(2);
    y0 << 1.0, 0.4;
    std::vector<var> rates{0.7 + 0.05 * i, 1.5, 0.9, 0.35};
    Eigen::Matrix<var, Eigen::Dynamic, 1> tau(2);
    tau << 1.1 - 0.02 * i, 0.65;
    auto ys = stan::math::dde_rk45_tol(nonlinear_rhs(), y0, 0.0, ts, tau, 1e-8,
                                       1e-8, 1000000, nullptr, rates, 0.2);
    var total = 0;
    for (const auto& y : ys) {
      total += y.sum();
    }
    total.grad();
    std::vector<double> result{total.val(),    y0(0).adj(),    y0(1).adj(),
                               rates[0].adj(), rates[1].adj(), rates[2].adj(),
                               rates[3].adj(), tau(0).adj(),   tau(1).adj()};
    stan::math::recover_memory();
    return result;
  };

  std::vector<std::vector<double>> sequential;
  for (int i = 0; i < num_threads; ++i) {
    sequential.push_back(solve_and_grad(i));
  }

  std::vector<std::vector<double>> concurrent(num_threads);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      stan::math::ChainableStack thread_instance;
      concurrent[i] = solve_and_grad(i);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    ASSERT_EQ(concurrent[i].size(), sequential[i].size());
    for (size_t k = 0; k < sequential[i].size(); ++k) {
      EXPECT_EQ(concurrent[i][k], sequential[i][k])
          << "thread " << i << ", entry " << k;
    }
  }
}
#endif
